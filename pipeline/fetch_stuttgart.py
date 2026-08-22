"""Fetch the HdM-HDR-2014 "Stuttgart HDR" dataset onto the research share.

Downloads via the dataset's public FTP (credentials as published on the
dataset page) with resume support, then writes a _dataset_info.json so
scan_sources / prepare_pairs can pick the footage up on the next run.

    python pipeline\\fetch_stuttgart.py --list
    python pipeline\\fetch_stuttgart.py                      # graded Rec.2020 TIFF (default)
    python pipeline\\fetch_stuttgart.py --set camera_exr     # ALEXA-WG scene-linear EXR
    python pipeline\\fetch_stuttgart.py --only Beerfest Carousel   # subset by name

Which set to take:
  - graded_tiff (DEFAULT, ~189 GB): Rec.2020 primaries, 0.005-4000 cd/m2 —
    matches RUDRA's working convention directly, no primaries conversion.
  - camera_exr (~80 GB): scene radiance in ALEXA Wide Gamut — smaller, but
    the ingest pipeline does NOT convert primaries; only use this if you add
    an AWG->Rec.2020 step (rudra.delivery.colorspace has the math).

LICENSE: free for academic/educational use; COMMERCIAL use requires a
licensing agreement with HdM Stuttgart (contact on the dataset page).
Clear this before shipping any model trained on it commercially.

Runs on plain CPython (stdlib only). Safe to re-run: finished files are
skipped by size, partial files resume via FTP REST.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from ftplib import FTP, error_perm, error_temp
from pathlib import Path

HOST = "hdr-2014.hdm-stuttgart.de"
USER = "HdM-HDR-2014"
PASSWORD = "Ht%eW84%p="

SETS = {
    "graded_tiff": "HDR_Color_Graded",
    "camera_exr": "HDR_Camera_Footage",
    "dcp": "Wide_Color_Gamut_Comparison_DCP",
}
DEFAULT_DEST = r"\\192.168.100.200\Data\08_Research\Source_HDR\Stuttgart_HDR_2014"


def connect() -> FTP:
    ftp = FTP(HOST, timeout=60)
    ftp.login(USER, PASSWORD)
    ftp.set_pasv(True)
    # binary mode up front: some servers refuse SIZE in ASCII mode, which
    # would make the NLST fallback misclassify files as directories
    ftp.voidcmd("TYPE I")
    return ftp


def list_dir(ftp: FTP, path: str) -> tuple[list[str], list[tuple[str, int]]]:
    """Return (subdir names, [(file name, size)]) for one directory."""
    dirs: list[str] = []
    files: list[tuple[str, int]] = []
    try:
        for name, facts in ftp.mlsd(path or "/"):
            if name in (".", ".."):
                continue
            if facts.get("type") == "dir":
                dirs.append(name)
            elif facts.get("type") == "file":
                files.append((name, int(facts.get("size", 0))))
        return dirs, files
    except error_perm:
        pass  # server without MLSD support -> NLST + SIZE probing
    for entry in ftp.nlst(path or ""):
        name = entry.rsplit("/", 1)[-1]
        full = f"{path}/{name}" if path else name
        try:
            size = ftp.size(full)
        except (error_perm, error_temp):
            size = None
        if size is None:
            dirs.append(name)
        else:
            files.append((name, size))
    return dirs, files


# keyword fallbacks used when the published folder names don't match the server
_SET_KEYWORDS = {
    "graded_tiff": ("grad", "color", "tiff", "rec2020"),
    "camera_exr": ("camera", "footage", "exr", "raw"),
    "dcp": ("dcp", "gamut", "comparison"),
}


def resolve_root(ftp: FTP, set_name: str) -> str:
    """Find the real top-level folder for a set; the published names have
    drifted from the server at least once (550 on HDR_Color_Graded)."""
    root_dirs, _ = list_dir(ftp, "")
    wanted = SETS[set_name]
    for d in root_dirs:
        if d.lower() == wanted.lower():
            return d
    for d in root_dirs:
        if any(k in d.lower() for k in _SET_KEYWORDS[set_name]):
            print(f'note: using server folder "{d}" for set {set_name!r} '
                  f'(published name "{wanted}" not found)')
            return d
    listing = "\n  ".join(root_dirs) or "(empty)"
    raise SystemExit(
        f'No folder matching set {set_name!r} on the server.\n'
        f"Server root contains:\n  {listing}\n"
        f"Rerun with --root <one of the above>.")


_JUNK = (".ds_store", "thumbs.db", "desktop.ini")


def _is_junk(name: str) -> bool:
    low = name.lower()
    return low in _JUNK or name.startswith("._")


def walk(ftp: FTP, root: str):
    """Yield (remote_path, size) for every file under root, recursively.
    macOS/Windows metadata junk on the server is skipped."""
    pending = [root]
    while pending:
        cur = pending.pop()
        subdirs, files = list_dir(ftp, cur)
        for name in subdirs:
            if not _is_junk(name):
                pending.append(f"{cur}/{name}")
        for name, size in files:
            if not _is_junk(name):
                yield f"{cur}/{name}", size


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download(ftp_factory, files, remote_root: str, dest: Path) -> tuple[int, int]:
    done = skipped = 0
    total = len(files)
    ftp = ftp_factory()
    for i, (remote, size) in enumerate(files, 1):
        rel = remote[len(remote_root):].lstrip("/")
        target = dest / rel.replace("/", "\\") if "\\" in str(dest) else dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        have = target.stat().st_size if target.exists() else 0
        if size and have == size:
            skipped += 1
            continue
        mode = "ab" if 0 < have < (size or 0) else "wb"
        offset = have if mode == "ab" else 0
        for attempt in range(5):
            try:
                with open(target, mode) as fh:
                    ftp.retrbinary(f"RETR {remote}", fh.write, blocksize=1 << 20,
                                   rest=offset or None)
                done += 1
                print(f"[{i}/{total}] {rel}  ({human(size or target.stat().st_size)})")
                break
            except (error_temp, EOFError, OSError, ConnectionError) as exc:
                print(f"  retry {attempt + 1}/5 after error: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
                try:
                    ftp.quit()
                except Exception:
                    pass
                ftp = ftp_factory()
                have = target.stat().st_size if target.exists() else 0
                mode, offset = ("ab", have) if have else ("wb", 0)
        else:
            print(f"  FAILED after 5 attempts: {remote}", file=sys.stderr)
    try:
        ftp.quit()
    except Exception:
        pass
    return done, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=sorted(SETS), default="graded_tiff")
    parser.add_argument("--root", default=None,
                        help="exact server folder (overrides --set name resolution)")
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--only", nargs="*", default=None,
                        help="substring filters on sequence/folder names")
    parser.add_argument("--list", action="store_true", help="inventory only, no download")
    parser.add_argument("--list-root", action="store_true",
                        help="print the server's top-level folders and exit")
    args = parser.parse_args()

    print(f"Connecting to ftp://{HOST} ...")
    ftp = connect()
    if args.list_root:
        root_dirs, root_files = list_dir(ftp, "")
        print("Server root:")
        for d in sorted(root_dirs):
            print(f"  [dir]  {d}")
        for name, size in sorted(root_files):
            print(f"  [file] {name}  ({human(size)})")
        return 0
    remote_root = args.root or resolve_root(ftp, args.set)
    print(f"Scanning {remote_root} ...")
    files = sorted(walk(ftp, remote_root))
    try:
        ftp.quit()
    except Exception:
        pass

    if args.only:
        needles = [n.lower() for n in args.only]
        files = [f for f in files if any(n in f[0].lower() for n in needles)]
    total_bytes = sum(s for _, s in files)
    seq_dirs = sorted({f[0][len(remote_root):].lstrip("/").split("/")[0] for f in files})
    print(f"{len(files)} files, {human(total_bytes)}, {len(seq_dirs)} top-level sequences:")
    for s in seq_dirs:
        print(f"  {s}")
    if args.list:
        return 0
    if not files:
        print("Nothing matched.", file=sys.stderr)
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    done, skipped = download(connect, files, remote_root, dest)
    info = {
        "dataset": "HdM-HDR-2014 (Stuttgart HDR)",
        "source": f"ftp://{HOST}/{remote_root}",
        "set": args.set,
        "convention": ("Rec.2020, 0.005-4000 cd/m2, display-graded TIFF"
                        if args.set == "graded_tiff"
                        else "ALEXA Wide Gamut scene-linear EXR (needs AWG->Rec.2020 before ingest)"),
        "license": "Academic free; commercial use requires HdM Stuttgart license",
        "files": len(files), "downloaded": done, "skipped_existing": skipped,
        "total_bytes": total_bytes, "sequences": seq_dirs,
    }
    (dest / "_dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nDone: {done} downloaded, {skipped} already present."
          f"\nInfo sidecar: {dest / '_dataset_info.json'}"
          f"\nNext: pipeline\\run_v3_now.bat  (scan will pick the new sequences up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
