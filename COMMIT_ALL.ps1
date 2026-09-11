# RUDRA -- apply the paper rewrite, the documentation cull and the new Studio
# interface, in three commits.
#
#   powershell -ExecutionPolicy Bypass -File COMMIT_ALL.ps1
#
# This replaces COMMIT_PAPER.ps1, COMMIT_DOCS.ps1 and COMMIT_UI.ps1. Those three
# each ran the full suite AFTER the file removals they depend on, so the paper
# test could never pass: mdtotex.py was still on disk because the git rm that
# removes it was gated behind the test that fails while it is there. Order fixed
# here, once, for all three.
#
# Order: put the working tree in its final state, run the suite ONCE against
# that state, then make the three commits. Nothing is committed unless the whole
# suite is green.
#
# If the suite fails, the removals have happened but nothing is committed. Every
# removed file is still in HEAD, so this puts them all back:
#
#     git checkout -- .
#
# -Redo <sha> rewinds to a commit first, keeping every change staged and the
# working tree untouched, so a run that committed the wrong grouping can be
# redone cleanly. Nothing is lost: git reflog still has the old commits.
#
#     powershell -File COMMIT_ALL.ps1 -Redo 150a3ca
#
param([string]$Redo = "")

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

# Anything that stops the run AFTER the removals has to say how to undo them,
# not only the test failure. A script that deletes 22 files and then exits with
# a LaTeX error and no way back is worse than one that never deleted anything.
$script:removedAny = $false
$script:tempRoot = $env:TEMP
if (-not $script:tempRoot) { $script:tempRoot = [System.IO.Path]::GetTempPath() }
if (-not (Test-Path -LiteralPath $script:tempRoot)) {
    New-Item -ItemType Directory -Force -Path $script:tempRoot | Out-Null
}
function Bail {
    param([string]$Why)
    Write-Host ""
    Write-Host $Why -ForegroundColor Red
    Write-Host "NOTHING was committed." -ForegroundColor Red
    if ($script:removedAny) {
        Write-Host "Files were removed from the working tree. They are all still in" -ForegroundColor Red
        Write-Host "HEAD, so this puts every one of them back:" -ForegroundColor Red
        Write-Host "    git checkout -- ." -ForegroundColor Red
    }
    exit 1
}

$paperNew = @("paper/main.tex", "paper/abstract.tex", "paper/appendix.tex",
              "paper/sec1_introduction.tex", "paper/sec2_related.tex",
              "paper/sec3_formulation.tex", "paper/sec4_method.tex",
              "paper/sec5_setup.tex", "paper/sec6_results.tex",
              "paper/sec7_failure.tex", "paper/sec8_bound.tex",
              "paper/sec9_gate.tex", "paper/sec10_methodology.tex",
              "paper/sec11_limitations.tex", "paper/sec12_conclusion.tex",
              "paper/refs.bib", "paper/main.bbl", "paper/build.sh",
              "paper/mkarxiv.sh", "paper/.gitignore", "paper/ABSTRACT_ARXIV.txt",
              "paper/main.pdf",
              # The mirror the repo front page links, written from the build
              # above. A test asserts the two files are byte-identical, so if
              # this is not committed alongside main.pdf the check passes in
              # the working tree and fails for everyone who clones.
              "research/RUDRA_HDR_2026.pdf")
$paperGone = @("paper/_body.tex", "paper/_abstract.tex", "paper/mdtotex.py")

$notesGone = @(
    "AUDIT_2026-07-15.md", "AUDIT_2026-08-10.md", "DECODER_CHEATSHEET.md",
    "DELIVERY_2026-08-22.md", "FIXES_2026-08-10.md", "MOAT_REVIEW_2026-08-22.md",
    "NEXT_2026-09-03.md", "PAPER_DRAFT_2026-08-29.md", "PAPER_ERRATA.md",
    "PAPER_RESULTS_TEMPLATE.md", "RUDRA_FIXED_TRAINING_GUIDE_2026-07-14.md",
    "RUDRA_HDR_IMAGE_VIDEO_AUDIT_2026-07-14.md", "RUDRA_RESEARCH_FINDINGS_2026-07-03.md",
    "RUDRA_RETRAIN_RUNBOOK.md", "RUDRA_TECHNICAL_REVIEW.md", "SDR2HDR_V2_STATUS.md",
    "TRAINING_GUIDE_2026-07-15.md", "TRAINING_STEPS_2026-08-10.md", "V02_PLAN_2026-09-04.md")

$docsNew = @("README.md", "STATUS.md", ".gitignore",
             "docs/RESULTS.md", "docs/TRAINING.md", "docs/INTERNALS.md")
$uiNew   = @("ui/index.html", "ui/style.css", "ui/shell.js",
             "ui/docs/studio_v2_mockup.html",
             "tests/ui_smoke/press_everything.py",
             "tests/test_committed_checkpoint_2026_09_03.py")
$corpusNew = @("rudra/sdr2hdr.py",
               "docs/CORPUS.md",
               "training/prepare_training_data.py",
               "training/pilot_clipping.py",
               "training/survey_datasets.py",
               "tests/test_corpus_convention_2026_09_11.py")

# The helper scripts are tracked, so leaving them modified blocks the rebase
# with "uncommitted changes to tracked files" and nothing can be pushed.
$scriptsNew = @("COMMIT_ALL.ps1", "PUSH_SYNC.ps1")
$scriptsGone = @("COMMIT_PAPER.ps1", "COMMIT_DOCS.ps1", "COMMIT_UI.ps1")

$needed = $paperNew + $docsNew + $uiNew + $corpusNew
$missing = $needed | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($missing) {
    Write-Host "ERROR: these are not on disk:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "   $_" }
    exit 1
}

if ($Redo) {
    Write-Host ""
    Write-Host "Rewinding to $Redo (changes kept, working tree untouched)." -ForegroundColor Cyan
    git reset --soft $Redo
    if ($LASTEXITCODE -ne 0) { Write-Host "reset failed; nothing changed" -ForegroundColor Red; exit 1 }
    git reset | Out-Null     # unstage as well, so each group stages only its own
    Write-Host "at $(git --no-optional-locks log --oneline -1)" -ForegroundColor Green
}

# ---- 1. working tree into its final state -------------------------------
Write-Host ""
Write-Host "Removing superseded files." -ForegroundColor Cyan
$removed = 0
foreach ($p in ($paperGone + $notesGone + $scriptsGone)) {
    if (Test-Path -LiteralPath $p) {
        Remove-Item -LiteralPath $p -Force
        if (Test-Path -LiteralPath $p) { Write-Host "  ! could not remove $p" -ForegroundColor Yellow }
        else { $removed++; $script:removedAny = $true; Write-Host "  - $p" }
    }
}
Write-Host "removed $removed." -ForegroundColor Green

# Rebuild the paper where there is a toolchain; otherwise the committed
# main.pdf is the one built alongside these sources and is used as it stands.
if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host ""
    Write-Host "Building the paper." -ForegroundColor Cyan
    Push-Location paper
    pdflatex -interaction=nonstopmode main.tex | Out-Null
    bibtex main | Out-Null
    pdflatex -interaction=nonstopmode main.tex | Out-Null
    pdflatex -interaction=nonstopmode main.tex | Out-Null
    $bad = Select-String -Path main.log -Pattern '^!'
    $undef = Select-String -Path main.log -Pattern 'Warning: (Reference|Citation)'
    Pop-Location
    if ($bad) { $bad | ForEach-Object { Write-Host $_.Line }; Bail "The paper did not build." }
    if ($undef) { $undef | ForEach-Object { Write-Host $_.Line }; Bail "The paper has undefined references." }
    Write-Host "built." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "No pdflatex here, so paper/main.pdf is used as delivered." -ForegroundColor Yellow
}

# research/ is what the repo front page points at, and a test asserts the two
# files are byte-identical. This is the copy that keeps that true, and it has to
# happen BEFORE the suite runs, not after it.
New-Item -ItemType Directory -Force -Path research | Out-Null
Copy-Item -LiteralPath "paper\main.pdf" -Destination "research\RUDRA_HDR_2026.pdf" -Force
Write-Host "research/RUDRA_HDR_2026.pdf written from paper/main.pdf." -ForegroundColor Green

# ---- 2. one test run, against the final state ---------------------------
Write-Host ""
Write-Host "Tests, once, against the finished tree." -ForegroundColor Cyan
python -m pytest tests/ -q --basetemp="$(Join-Path $script:tempRoot 'rudra-pytest')"
if ($LASTEXITCODE -ne 0) { Bail "Tests failed." }

# ---- 3. three commits ----------------------------------------------------
# Everything git currently tracks, so a path that is neither on disk nor in the
# index can be dropped before `git add` sees it. Without this, running the
# script twice dies on "pathspec did not match any files" for a file the first
# run already removed, which looks like a failure and is not one.
$script:tracked = @{}
foreach ($t in (git --no-optional-locks ls-files)) { $script:tracked[$t] = $true }

function Commit-Group {
    param([string[]]$Paths, [string]$Message, [string]$Label)
    $live = $Paths | Where-Object { (Test-Path -LiteralPath $_) -or $script:tracked.ContainsKey($_) }
    if (-not $live) { Write-Host "nothing to do for $Label, skipping" -ForegroundColor Yellow; return }
    # Empty the index first. `git commit` with no pathspec commits the WHOLE
    # index, so anything staged by an earlier run or by hand rides along: the
    # first version of this script put pilot_clipping.json and three corpus
    # files into a commit called "Rewrite the paper as a paper".
    git reset --quiet
    # -f because docs/*.md were ignored by a blanket *.md rule for months. The
    # rule is fixed in this commit; the force is what lets the commit that
    # fixes it also carry the files it un-ignores.
    git add -A -f -- $live
    if ($LASTEXITCODE -ne 0) { Bail "git add failed for $Label." }
    $staged = git --no-optional-locks diff --cached --name-only
    if (-not $staged) { Write-Host "nothing staged for $Label, skipping" -ForegroundColor Yellow; return }
    $f = Join-Path $script:tempRoot ("rudra_msg_" + $Label + ".txt")
    $Message | Set-Content -LiteralPath $f -Encoding ASCII
    git commit -F $f
    if ($LASTEXITCODE -ne 0) { Bail "git commit failed for $Label." }
    Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
    Write-Host "committed: $Label" -ForegroundColor Green
}

$msgPaper = @"
Rewrite the paper as a paper

The content was sound and the form was not. It opened with a withdrawal notice
instead of an introduction, stated no problem formally, carried no equations,
and its tables were unnumbered pandoc longtables with the captions faked as bold
body text. Sections were referred to as "S6" by hand because there were no
labels to reference.

No claim and no number changed in the restructuring. Every measured value in the
old source appears in the new one; the only numeric tokens that do not carry over
are hand-written section numbers and a LaTeX column width, and the only ones
added are model constants now stated explicitly in the method, two corpus
percentages from the loss docstring, and the standards' numbers.

  * An introduction with six numbered contributions, and the withdrawal of the
    earlier manuscript moved to an appendix.
  * A problem formulation: the forward map and the two distinct ways it fails to
    be injective, the censored-observation treatment written out, the units
    convention fixed once because conflating the two is a multi-stop error that
    produces plausible-looking numbers.
  * A method section with the composite, the per-pixel prior and the full
    training objective as equations taken from rudra/sdr2hdr.py.
  * 18 numbered tables and 3 numbered figures, captioned and cross-referenced.
    The build now fails on an undefined reference: it renders as a bold ?? that
    a skim reads straight past.
  * Order follows the argument: localise the failure, measure the bound, then
    build the component that reaches the reachable part of it.
  * A statistical protocol paragraph saying what is paired, what n is, and why
    three seeds support a sign and not an interval.
  * refs.bib and natbib in place of a hand-maintained thebibliography.

Two claims were then weakened after review, and both were overclaims of the same
kind. "97% of the signal is absent from an 8-bit frame" is not what was measured:
a linear readout of one head's pooled features explains 3% of the oracle scale's
variance on 102 samples, which bounds that readout and that representation and
not the frame. Section 8.2 now separates what is established (the 79/21 variance
split caps any condition-detection architecture at R^2 0.213, and that holds for
any architecture of that shape) from what is measured of one representation from
what is not established at all. And "the two metrics disagree by two orders of
magnitude" compares decibels with JOD, which are different units and not
commensurable as numbers; the claim is now about the practical significance each
assigns to the same difference, which is what was actually observed.

Two limitations the review found are now stated rather than implied: the
analytic baseline is the exact inverse of the transform that rendered the corpus,
so on clean frames it is being handed the forward pipeline; and the gate is
supervised on our own degradation label, so what it learns may be this
generator's signature rather than "these shadows need reconstruction", which
nothing in the evaluation separates.

The markdown-to-LaTeX pipeline is gone. mdtotex.py and the generated _body.tex
and _abstract.tex are removed and the LaTeX is the only source. Markdown cannot
express an equation, a numbered float or a cross-reference, and keeping a
generator meant a clone without the markdown built a different document from the
one the author was editing.
"@

$msgDocs = @"
One README, one STATUS, three guides, and a paper

Nineteen dated markdown files sat in the repository root: audits from July and
August, two superseded training guides, a retrain runbook, review findings, a
decoder cheatsheet, a moat review, three planning documents, the earlier paper
draft and its errata. Together they were a record of how the project got here
rather than of what it does, and nothing told a reader which of them were still
true. The paper's errata file is the clearest case: the corrections it demanded
have all landed, and the withdrawal it describes is now an appendix of the paper.

Kept deliberately, because each is read by something rather than by nobody:

  AGENTS.md               the rule about never running git from a mounted
                          shell, which has cost a week of uncommitted work on
                          four separate occasions when it was not followed
  STATUS.md               the only document that keeps the three RUDRAs apart
  HF_MODEL_CARD.md        the HuggingFace upload reads it
  docs/HUB_MODEL_CARD.md  a test reads it
  checkpoints/README.md   the model registry and the weights licence

The README gains a table saying where everything is written down, and repeats
neither of the two claims the paper walked back. Two dangling references in
STATUS.md, to PAPER_ERRATA.md and RUDRA_TECHNICAL_REVIEW.md, are rewritten
rather than left to rot.

docs/RESULTS.md, docs/TRAINING.md and docs/INTERNALS.md are added explicitly.
They were split out of the README earlier and their links 404'd on github
because the split was never committed, which is why the links were removed
there. The files are in this commit and the links are back.
"@

$msgUi = @"
Rebuild the Studio interface around the same controls

The application was not the problem. app.js, compositor.js and server.py are
untouched by this commit. What changed is the shell: the markup and the styling
that decide what an artist sees first, what they have to hunt for, and how much
of the window the picture gets. Every one of the 56 element ids, the eight-menu
structure and the eight classes the application toggles survive, so every
existing behaviour runs through exactly the code path it ran through before.

The viewer gets a toolbar of its own. What the picture is being shown as,
compare mode, framing guides and zoom now sit above the image, which is where
Nuke, Resolve and RV put them. RUDRA, Baseline and Wipe moved out of the
transport bar into that toolbar, because they are viewing controls and not
playback.

The inspector is three tabs instead of one 700-pixel column: Reconstruct, Grade,
Deliver. A menu item whose control lives on a tab that is not showing brings its
tab forward, because a control that changes something invisible is a control the
user will believe did nothing.

Two workspaces. Simple hides the scopes dock, the log and the explanatory notes
and leaves open, look, compare, render. Full is everything. Simple hides nothing
that is the only way to reach a behaviour: every control it removes is also on a
menu, which is the line between a simplified interface and a crippled one.

An icon rail toggles the three panels, and the drop zone tucks into one line
once something is open, because a permanent invitation to open a file, sitting
above the list of files you already opened, is how a left rail turns into wasted
rail.

ui/shell.js is the new chrome and owns no state that app.js owns. Where a shell
control does something the application already does, it clicks the existing
control rather than reimplementing it, so there is one code path per action and
the two cannot drift. It also runs if app.js failed.

One real bug, found by driving the new page in headless Chromium before any of
this was committed: the first draft of the stylesheet set display on .overlay
without restoring the [hidden] rule, and an invisible full-screen overlay ate
every click in the application with nothing on screen to explain why. The old
stylesheet had a comment about exactly that. The fix is now a blanket
[hidden]{display:none !important} rather than one rule for one element.

Deliberately NOT in this commit, because each needs work behind the UI and
shipping the control before the behaviour would put decoration in a tool whose
whole argument is that it measures what it claims: a cursor probe reading nits,
per-frame cache state on the timeline, ProRes/HDR10/HLG targets with a render
queue, and split and difference compare modes. The Deliver tab says so on its
face rather than showing a button that does nothing.

The interface smoke test gains three checks and one fix: it clicks through to
the Grade tab before dragging a region, because a panel that is not showing has
no layout and bounding_box() returns None, which reads as a broken control
rather than a hidden one.

docs/rudra_studio.png still shows the old interface. Recapture it with
ui/capture_shot.py once a plate is loaded.
"@

$msgCorpus = @"
Render a corpus that contains the thing the model is trained to reconstruct

Two defects in training/prepare_training_data.py, both measured, the second
found while testing the first.

1. THE SDR SIDE DID NOT CLIP

The render applied -1 EV before the ACES curve, commented "slight underexpose
for safety". The ACES approximation saturates near a scene-linear input of 7.24,
so halving the input roughly doubles the radiance a pixel needs before it blows.
On the corpus that produced: median clipped fraction 0.000%, and 52.6% of frames
with no clipped pixel anywhere.

An inverse tone mapper is a machine for saying what was above a blown highlight.
A training set without blown highlights does not contain the question.

2. THE TOP CODE WAS UNREACHABLE

oetf_srgb(1.0) is 1.055 * 1**(1/2.4) - 0.055, which in float32 lands on
0.99999994. Times 255 that is 254.99998, and astype(uint8) TRUNCATED it to 254.

So the corpus SDR could not contain the value 255 at any exposure. Every fully
blown pixel was stored one code below full. Anything testing for clipping by
== 255 found none, ever, which is part of why the first defect went unseen. And
the model never saw the top code in training while real delivered SDR is full of
it: train/serve skew at exactly the pixels this model exists for.

Both quantisation steps round now. On a scene-linear ramp to 16x diffuse white:

    -1 EV, truncating (shipped)   max code 251   clipped 0.000
    -1 EV, rounding               max code 255   clipped 0.203
     0 EV, rounding               max code 255   clipped 0.601

THE CONVENTION IS NOW CARRIED, NOT REMEMBERED

sdr_to_baseline_hdr took a literal 2 * 203/10000. The 2 was not a constant of
nature, it was 2**(-(-1 EV)): the render's exposure, undone. Written down it was
correct and unexplained, and it would have become silently wrong the moment the
render changed, which is what this commit does.

The scale is derived now, sdr_to_baseline_hdr(sdr, corpus_ev) takes the offset,
and a test asserts the default is bit-for-bit what it replaced. SDR2HDRNet
carries corpus_ev and uses it for its own baseline; from_config reads it; a
checkpoint whose config omits it is legacy, which is correct for every
checkpoint that exists. --tonemap-ev sets it at render time and every sidecar
records both the exposure used and the resulting clipped_fraction, so the next
corpus states on its face whether it contains the phenomenon. That measurement
not existing is why this ran for months.

WHAT TO DO WITH IT

training/pilot_clipping.py samples real source frames, tone-maps each at both
conventions and reports the two numbers the audit reported, exiting non-zero if
the new one still does not clip. The full render is hours; this says in minutes
whether it is worth starting, and on low-dynamic-range sources it fails and says
why: a set whose ground truth peaks near diffuse white has no highlights to clip
in the first place.

training/survey_datasets.py walks the dataset roots and the manifests and
reports what is still referenced, including the corpus SHA pinned in each
checkpoint config. It proposes and never deletes, because a result whose corpus
has been deleted stops being reproducible the moment you need to defend it.

docs/CORPUS.md is the specification: eight classes counted in independent
scenes rather than frames, a measured dynamic-range floor for each, the
per-scene record a scene needs before it enters a manifest, and the licence
position. It also records what the inventories on E: actually say, which is
worse than expected: the local set's ground truth has a median dynamic range of
3.97 stops and no source above 10 000 nits, two of the three sets have none at
all, and the video manifest is gated at two independent training scenes.

Nothing here retrains anything. The order is render, retrain, remeasure, one
change at a time, because the paper reports corpus size as a lever that moved
the bound 0.03 dB and this is a claim about corpus CONTENT instead.
"@

$msgScripts = @"
One commit script, and the gitignore rule that caused the 404s

COMMIT_PAPER.ps1, COMMIT_DOCS.ps1 and COMMIT_UI.ps1 each ran the full test
suite AFTER the file removals those tests assert on, so the paper test could
never pass: mdtotex.py stayed on disk because the git rm that removes it was
gated behind the test that fails while it is there. COMMIT_ALL.ps1 replaces all
three: working tree into its final state, suite once against that state, then
the commits.

Two faults it had of its own, both found by running it rather than reading it:

  * `git commit` with no pathspec commits the whole index, so anything staged
    by an earlier run rode along. A commit titled "Rewrite the paper as a
    paper" contained pilot_clipping.json and three corpus files. Each group now
    empties the index before staging its own.
  * The helper scripts are tracked, so leaving them modified blocked the rebase
    with "uncommitted changes to tracked files" and nothing could be pushed.
    They are committed here.

And the one that mattered most. .gitignore had a blanket `*.md` with a single
negation for the root README. That silently ignored docs/RESULTS.md,
docs/TRAINING.md and docs/INTERNALS.md, so the guides split out of the README
were never committed, their links 404'd for everyone who clicked them, and a
commit went in on the remote removing the LINKS rather than adding the FILES.
An ignored file is not an error, so nothing reported it.

docs/*.md, STATUS.md and HF_MODEL_CARD.md are now negated explicitly. The
blanket rule stays for what it was written for: dated audits, runbooks and
session notes do not belong in the repo.
"@

Commit-Group -Paths ($paperNew + $paperGone) -Message $msgPaper -Label "paper"
Commit-Group -Paths ($docsNew + $notesGone)  -Message $msgDocs  -Label "docs"
Commit-Group -Paths $uiNew                   -Message $msgUi    -Label "ui"
Commit-Group -Paths $corpusNew               -Message $msgCorpus -Label "corpus"
Commit-Group -Paths ($scriptsNew + $scriptsGone) -Message $msgScripts -Label "scripts"

# ---- 4. leave the index exactly as HEAD ----------------------------------
# Commit-Group empties the index BEFORE each `git add`, never after the last
# commit. So anything staged outside the five path lists -- HF_MODEL_CARD.md,
# newly un-ignored by the .gitignore fix, is the real case -- survives the whole
# run as a staged addition nobody committed. Nothing notices until the rebase
# refuses to start with "uncommitted changes to tracked files", three scripts
# later, where it looks like a git problem rather than this one.
git reset --quiet
$leftover = git --no-optional-locks status --porcelain --untracked-files=no
if ($leftover) {
    Write-Host ""
    Write-Host "Tracked files still differ from HEAD after the five commits:" -ForegroundColor Yellow
    $leftover | ForEach-Object { Write-Host "   $_" }
    Write-Host "The rebase will refuse to start until these are committed." -ForegroundColor Yellow
}

# Files the .gitignore fix un-ignores but no path list claims. ls-files
# --others --exclude-standard reads the NEW .gitignore, so this is the exact
# set the blanket *.md rule was hiding and the negations now expose.
$unignored = git --no-optional-locks ls-files --others --exclude-standard
if ($unignored) {
    Write-Host ""
    Write-Host "Un-ignored but untracked -- decide whether these belong in the repo:" -ForegroundColor Yellow
    $unignored | ForEach-Object { Write-Host "   $_" }
}

Write-Host ""
Write-Host "Five commits made locally. NOT pushed." -ForegroundColor Green
Write-Host "The branch is still one behind origin and the README rebase is" -ForegroundColor Cyan
Write-Host "unresolved. Sort that out with PUSH_SYNC.ps1 before pushing." -ForegroundColor Cyan
Write-Host ""
Write-Host "The UI smoke test is not in the pytest suite: it needs the Studio" -ForegroundColor Yellow
Write-Host "running. It presses all 56 controls and it is the check that matters" -ForegroundColor Yellow
Write-Host "for the interface commit:" -ForegroundColor Yellow
Write-Host "    python ui\server.py" -ForegroundColor Yellow
Write-Host "    python tests\ui_smoke\press_everything.py" -ForegroundColor Yellow
Write-Host ""
git --no-optional-locks log --oneline -6
