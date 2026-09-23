"""N7: rudra-studio trains only on sources that may train sold weights."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.licences import classify  # noqa: E402


def test_real_v4b_paths_classify():
    cases = {
        "E:/source_hdr/Stuttgart_HDR_2014/carousel_fireworks::carousel_fireworks": ("hdm", False),
        r"G:\datasets\sources\hdm_hfr_2017\Bar\Bar_0001::Bar": ("hdm", False),
        "G:/datasets_rudra/netflix_sparks/ACES::SPARKS_ACES": ("netflix_sparks", True),
        "G:/datasets/sources/netflix_chimera/tif_DCI4k2398p::chimera_0001": ("netflix_chimera", True),
        "E:/RUDRA_v3/polyhaven/abandoned_bakery_2k::abandoned_bakery_2k": ("polyhaven", True),
        "G:/RUDRA_v02/pairs_moves/sdr::move_0001": ("polyhaven_moves", True),
        "D:/somewhere/mystery::clip": ("unknown", False),
        "E:/source_hdr/PolyHaven::stuttgart_hillside_2k": ("polyhaven", True),
    }
    for scene, (sid, ok) in cases.items():
        got = classify({"scene_id": scene})
        assert (got["source"], got["commercial_ok"]) == (sid, ok), (scene, got)


def test_commercial_only_drops_hdm_and_unknown(tmp_path, monkeypatch):
    from pipeline import build_manifests as bm
    rows = []
    for i, root in enumerate(["E:/source_hdr/Stuttgart_HDR_2014/a", "G:/x/netflix_chimera/b",
                              "G:/x/polyhaven/c", "D:/nobody/d"]):
        for j in range(4):
            rows.append({"asset_id": f"{i}_{j}", "scene_id": f"{root}::s{i}{j}", "is_video": False,
                         "peak_nits": 1000.0, "sdr_path": f"{root}/{j}.png",
                         "hdr_path": f"{root}/{j}.png", "tonemap_ev": 0.0,
                         "sdr_clipped_fraction": 0.03})
    (tmp_path / "pairs_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "m"
    monkeypatch.setattr(sys, "argv", ["build_manifests.py", "--pairs-dir", str(tmp_path),
                                      "--out-dir", str(out), "--commercial-only",
                                      "--min-video-share", "0", "--allow-problems"])
    bm.main()
    manifest = [json.loads(l) for l in (out / "sdr_hdr_manifest.jsonl").read_text().splitlines()]
    assert manifest and all(r["commercial_ok"] for r in manifest)
    assert {r["licence_source"] for r in manifest} == {"netflix_chimera", "polyhaven"}


def test_aces_ap0_sources_decode_into_rec2020():
    import numpy as np
    from training.prepare_training_data import to_scene_linear
    from rudra.delivery.colorspace import rgb_to_rgb_matrix
    grey = np.full((2, 2, 3), 0.18, np.float32)
    k = 100.0 / 203.0                          # ACES 1.0 = 100 nits, corpus 1.0 = 203
    assert np.allclose(to_scene_linear(grey, "aces"), 0.18 * k, atol=2e-3)   # neutral stays neutral
    red = np.zeros((1, 1, 3), np.float32); red[..., 0] = 1.0
    expected = rgb_to_rgb_matrix("ap0", "rec2020") @ np.array([1.0, 0, 0])
    assert np.allclose(to_scene_linear(red, "aces")[0, 0], np.clip(expected, 0, None) * k, atol=1e-3)


def test_paired_gate_reads_bench_csvs(tmp_path):
    from training.paired_gate import compare, load
    rows = "clip,frame,pu_psnr_db,cvvdp_jod\n" + "".join(f"c,f{i},{30 + i % 3},{8.0}\n" for i in range(30))
    better = "clip,frame,pu_psnr_db,cvvdp_jod\n" + "".join(f"c,f{i},{31 + i % 3},{8.1}\n" for i in range(30))
    (tmp_path / "a.csv").write_text(better); (tmp_path / "b.csv").write_text(rows)
    r = compare(load(tmp_path / "a.csv"), load(tmp_path / "b.csv"))
    assert r["pu_psnr_db"]["ci95"][0] > 0.9 and r["cvvdp_jod"]["wins"] == 30


def test_inventory_paths_follow_the_migration_to_g(tmp_path):
    """23 Sep 2026: the v4c ingest skipped every HdM-HFR frame because the v4b
    inventory still said E:\\source_hdr\\HdM-HFR-2017_Color-Graded, renamed to
    G:\\datasets\\sources\\hdm_hfr_2017 by the migration."""
    from pipeline.reclassify_inventory import MIGRATIONS, main, resolve
    have = {r"G:\datasets\sources\hdm_hfr_2017\Bar\A\A.00578888.tif",
            r"G:\datasets\sources\stuttgart_hdr_2014\carousel_fireworks\c.tif"}
    ex = have.__contains__
    assert resolve(r"E:\source_hdr\HdM-HFR-2017_Color-Graded\Bar\A\A.00578888.tif",
                   MIGRATIONS, ex) == r"G:\datasets\sources\hdm_hfr_2017\Bar\A\A.00578888.tif"
    assert resolve("E:/source_hdr/Stuttgart_HDR_2014/carousel_fireworks/c.tif", MIGRATIONS, ex) \
        == r"G:\datasets\sources\stuttgart_hdr_2014\carousel_fireworks\c.tif"
    assert resolve(r"E:\nowhere\x.tif", MIGRATIONS, ex) is None
    # a real file on disk is kept as is, and a missing one fails the run
    real = tmp_path / "real.exr"; real.write_bytes(b"x")
    src = tmp_path / "inv.jsonl"
    src.write_text(json.dumps({"path": str(real), "kind": "image", "scene_id": "s::a"}) + "\n"
                   + json.dumps({"path": r"E:\nowhere\x.tif", "kind": "image", "scene_id": "s::b"}) + "\n")
    assert main(src, tmp_path / "out.jsonl") == 1
    assert main(src, tmp_path / "out.jsonl", allow_missing=True) == 0
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert rows[0]["path"] == str(real) and rows[0]["scene_id"] == "s::a"


def test_commercial_inventory_never_looks_for_hdm(tmp_path):
    from pipeline.reclassify_inventory import main
    real = tmp_path / "polyhaven" / "a.exr"; real.parent.mkdir(); real.write_bytes(b"x")
    src = tmp_path / "inv.jsonl"
    src.write_text(json.dumps({"path": str(real), "kind": "image", "scene_id": "polyhaven::a"}) + "\n"
                   + json.dumps({"path": r"E:\source_hdr\HdM-HFR-2017_Color-Graded\Bar\b.tif",
                                 "kind": "image", "scene_id": "E:/source_hdr/HdM-HFR-2017_Color-Graded/Bar::b"}) + "\n")
    assert main(src, tmp_path / "out.jsonl", commercial_only=True) == 0
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["scene_id"] for r in rows] == ["polyhaven::a"]


def test_g_drive_classification_fixes():
    """Found on G: 23 Sep 2026: Sparks ships PQ codes in EXRs (filename says
    so), and Poly Haven's venice_* scenes matched the Sony 'venice' keyword."""
    from pathlib import Path as P
    from pipeline.scan_sources import guess_encoding as g
    assert g(P("G:/datasets_rudra/netflix_sparks/SPARKS_P3_PQ_4000nit_13776.exr"))[0] == "pq"
    assert g(P("G:/datasets_rudra/netflix_sparks/SPARKS_ACES_00001.exr"))[0] == "aces"
    assert g(P("E:/source_hdr/PolyHaven/venice_sunrise_2k.exr"))[0] == "linear"
    assert g(P("G:/x/alexa35_logc4/a.exr"))[0] == "logc4"


def test_duplicate_sparks_rendition_is_dropped(tmp_path):
    from pipeline.reclassify_inventory import main
    a = tmp_path / "SPARKS_ACES_00001.exr"; a.write_bytes(b"x")
    b = tmp_path / "SPARKS_P3_PQ_4000nit_00001.exr"; b.write_bytes(b"x")
    src = tmp_path / "inv.jsonl"
    src.write_text("".join(json.dumps({"path": str(x), "kind": "image", "scene_id": f"s::{x.stem[:-6]}"}) + "\n"
                           for x in (a, b)))
    assert main(src, tmp_path / "out.jsonl") == 0
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [Path(r["path"]).name for r in rows] == ["SPARKS_ACES_00001.exr"]
