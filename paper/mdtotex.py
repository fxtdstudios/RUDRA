"""Regenerate _abstract.tex / _body.tex from PAPER_DRAFT_2026-08-29.md.

The markdown is the source of truth. main.tex is a stable wrapper that
\input{}s the two files this script writes, so editing the paper means
editing the markdown and re-running build.sh -- never editing the .tex.
"""
import io, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "PAPER_DRAFT_2026-08-29.md")

CITE = {1: "eilertsen2017", 2: "marnerides2018", 3: "endo2017", 4: "liu2020",
        5: "santos2020", 6: "pu21", 7: "cvvdp", 8: "froehlich2014",
        9: "kalantari2017"}

# longtables whose cells are too wide for l-columns; keyed by a unique
# string in the table's header row.
WIDE = {
    "v5 (shipped)": r"@{}lp{0.46\linewidth}p{0.20\linewidth}@{}",
    # SS5.1's method column carries full sentences of label.
    "RUDRA + gate, as deployed": r"@{}p{0.52\linewidth}rr@{}",
}

# Tables that fit only at a smaller size. Seven columns of signed decimals do
# not go into a 6.5in text block at 11pt, and shrinking the type is honest
# where dropping a column would not be.
SMALL = ("clean dB | clean JOD | hard dB | hard JOD", "seed & clean dB")


def pandoc(md, *extra):
    return subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", "--wrap=preserve",
         "--no-highlight", *extra],
        input=md, capture_output=True, text=True, check=True).stdout


def main():
    src = io.open(SRC, encoding="utf-8").read()
    body = src[:src.index("\n---\n\n## References")]
    body = body[body.index("## Abstract"):].replace("## Abstract\n", "", 1)

    for n, key in CITE.items():
        body = body.replace("[%d]" % n, r"\cite{%s}" % key)

    i = body.index("\n---\n")
    abstract, rest = body[:i].strip(), body[i:].lstrip("\n-").lstrip()

    # Vector figures. The markdown already carries a numbered prose caption
    # ("**Figure 1.** ...") under each image, so emit a bare float rather than
    # letting pandoc build a second, redundant \caption from the alt text.
    rest = re.sub(
        r"!\[[^\]]*\]\(docs/figures/([A-Za-z0-9_]+)\.png\)",
        lambda m: ("\n\\begin{figure}[htbp]\n\\centering\n"
                   "\\includegraphics[width=\\linewidth]{%s.pdf}\n"
                   "\\end{figure}\n" % m.group(1)),
        rest)

    # drop hand-written section numbers -- LaTeX numbers these itself and the
    # numbers come out identical, so every "SS6" cross-reference still holds
    rest = re.sub(r"^(#{2,4})\s+\d+(\.\d+)*\.?\s+", r"\1 ", rest, flags=re.M)

    tex = pandoc(rest, "--shift-heading-level-by=-1")
    tex = tex.replace(
        "\\begin{center}\\rule{0.5\\linewidth}{0.5pt}\\end{center}\n", "")
    tex = re.sub(r"\n{3,}", "\n\n", tex)

    # widen the tables that overflow the text block: the header row sits a
    # couple of lines below \\begin{longtable}, so look ahead rather than at
    # the next line only.
    lines = tex.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("\\begin{longtable}"):
            continue
        window = "\n".join(lines[i + 1:i + 6])
        for needle, cols in WIDE.items():
            if needle in window:
                lines[i] = re.sub(r"\{@\{\}.*\}$",
                                  lambda m, c=cols: "{" + c + "}", line)
                break
        if any(needle in window for needle in SMALL):
            lines[i] = "{\\footnotesize\n" + lines[i]
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("\\end{longtable}"):
                    lines[j] = lines[j] + "\n}"
                    break
    tex = "\n".join(lines)

    io.open(os.path.join(HERE, "_body.tex"), "w", encoding="utf-8").write(tex)
    io.open(os.path.join(HERE, "_abstract.tex"), "w",
            encoding="utf-8").write(pandoc(abstract))
    print("wrote _body.tex (%d bytes) and _abstract.tex (%d chars of abstract)"
          % (len(tex), len(abstract)))
    if len(abstract) > 1920:
        sys.stderr.write(
            "note: abstract is %d chars; arXiv's metadata field caps at 1920. "
            "The PDF is fine -- trim only the text pasted into the web form.\n"
            % len(abstract))


if __name__ == "__main__":
    main()
