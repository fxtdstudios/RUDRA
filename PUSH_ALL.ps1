# RUDRA -- cull the stray markdown, commit everything, test, push.
#   powershell -ExecutionPolicy Bypass -File PUSH_ALL.ps1
#   powershell -ExecutionPolicy Bypass -File PUSH_ALL.ps1 -DryRun
#
# THE CULL
#
# Nineteen dated working notes used to sit in the repo root: audits, runbooks,
# review findings, superseded training guides. .gitignore has a blanket *.md
# rule to stop them coming back, with explicit negations for the set that is
# meant to be there. But a file that is ALREADY TRACKED is not affected by
# .gitignore at all -- the rule only governs what gets added. So anything that
# went in before the rule is still on the remote and will stay there forever
# unless it is removed on purpose.
#
# This removes every tracked *.md outside the keep list, and prints each one
# before it goes. `git rm` stages a deletion; the content stays in history, so
# nothing here is unrecoverable.
param([switch]$DryRun)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

function Bail {
    param([string]$Why)
    Write-Host ""
    Write-Host $Why -ForegroundColor Red
    Write-Host "Nothing was pushed." -ForegroundColor Red
    exit 1
}

$before = (git --no-optional-locks rev-parse HEAD).Trim()
Write-Host "HEAD is $before" -ForegroundColor DarkGray
Write-Host "To undo every commit below:  git reset --hard $before" -ForegroundColor DarkGray

# ---- the keep list -------------------------------------------------------
# Everything here is either linked from the README's documents table, read by
# a tool, or is a package's own README. Everything else is a working note.
$keep = @(
    "README.md",
    "STATUS.md",
    "HF_MODEL_CARD.md",
    "checkpoints/README.md",
    "pipeline/README.md"
)
$keepPrefix = @("docs/", ".github/")

$allMd = @(git --no-optional-locks ls-files -- "*.md")
$doomed = $allMd | Where-Object {
    $f = $_
    (-not ($keep -contains $f)) -and
    (-not ($keepPrefix | Where-Object { $f.StartsWith($_) }))
}

Write-Host ""
Write-Host "Tracked markdown: $($allMd.Count) files" -ForegroundColor Cyan
$allMd | Where-Object { $doomed -notcontains $_ } |
    ForEach-Object { Write-Host "   keep    $_" -ForegroundColor DarkGray }
if ($doomed) {
    $doomed | ForEach-Object { Write-Host "   REMOVE  $_" -ForegroundColor Yellow }
} else {
    Write-Host "   nothing to remove -- the cull already happened" -ForegroundColor Green
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run. Nothing staged, nothing committed, nothing pushed." -ForegroundColor Cyan
    exit 0
}

# ---- tests, before anything is staged ------------------------------------
Write-Host ""
Write-Host "Tests." -ForegroundColor Cyan
$tempRoot = $env:TEMP
if (-not $tempRoot) { $tempRoot = [System.IO.Path]::GetTempPath() }
python -m pytest tests/ -q --basetemp="$(Join-Path $tempRoot 'rudra-pytest')"
if ($LASTEXITCODE -ne 0) { Bail "Tests fail. Not committing." }

$script:tracked = @{}
foreach ($f in (git --no-optional-locks ls-files)) { $script:tracked[$f] = $true }

function Commit-Group {
    param([string[]]$Paths, [string]$Message, [string]$Label)
    $live = $Paths | Where-Object {
        (Test-Path -LiteralPath $_) -or $script:tracked.ContainsKey($_)
    }
    if (-not $live) { Write-Host "nothing on disk for $Label, skipping" -ForegroundColor Yellow; return }
    git reset --quiet
    git add -A -f -- $live
    if ($LASTEXITCODE -ne 0) { Bail "git add failed for $Label." }
    $staged = git --no-optional-locks diff --cached --name-only
    if (-not $staged) {
        Write-Host "nothing changed for $Label, skipping" -ForegroundColor Yellow
        git reset --quiet; return
    }
    Write-Host ""
    Write-Host "staging for ${Label}:" -ForegroundColor Cyan
    $staged | ForEach-Object { Write-Host "   $_" }
    $f = Join-Path $tempRoot ("rudra_msg_" + $Label + ".txt")
    $Message | Set-Content -LiteralPath $f -Encoding ASCII
    git commit -F $f
    if ($LASTEXITCODE -ne 0) { Bail "git commit failed for $Label." }
    Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
    Write-Host "committed: $Label" -ForegroundColor Green
}

# ---- 1. the interface ----------------------------------------------------
Commit-Group -Paths @(
    "ui/app.js", "ui/compositor.js", "ui/index.html", "ui/theme.css",
    "ui/assets/rudra-mark.png", "ui/assets/rudra-mark-256.png",
    "tests/ui_smoke/press_everything.py",
    "docs/rudra_studio.png",
    "README.md"
) -Label "ui" -Message @"
Re-skin the Studio, play at frame rate, and give the viewer instruments

PLAYBACK. The transport was a 160 ms setInterval -- a 6.25 fps ceiling with no
relationship to the clip -- with a skip-the-beat-if-busy guard that dropped
time instead of catching up, and nothing prefetched: select() fetched a frame
only once the playhead had landed on it, so every beat paid a round trip and a
forward pass, in series. It is a clock now. requestAnimationFrame presents
whichever frame is due by elapsed wall-clock time at the fps the server already
reported and nothing read; a frame that is not warm is SKIPPED, not waited for,
which is the whole difference between real time and slow motion. A read-ahead
holds twelve frames in front of the playhead on three parallel controllers,
kept separate from `inflight` because aborting is right for a click and fatal
for a queue. Five seconds at 24 fps now presents 120 distinct frames where the
old loop managed 31.

VIEWER. Three layers on the two float targets that were already resident for
the wipe, so each costs one uniform and no recomposite: false colour by nits
with a legend, |RUDRA - baseline| on a log ramp, and a per-pixel probe that
reports what the baseline had, what the network put there, the delta in stops,
the masks, and whether the SDR clipped at that point at all -- a lift where it
never clipped is invention rather than reconstruction, and nothing else in the
tool separates the two. Zoom and pan are a transform on the plate rather than a
canvas resize, so nothing re-rasterises and every client-to-image mapping stays
correct for free.

The probe's y is NOT flipped, and the first version of it was. COMPOSITE writes
through a pass-through quad, so inside those targets the image is already in
the framebuffer's bottom-up order; the flip belongs in DISPLAY, where it was.
Doubling it reads the mirrored scanline -- a bug that looks correct in every
vertically symmetric frame.

SCOPES. Both now carry one hue per zone, blue below four stops under diffuse
white to gold above two over, both mark 203 nits, the waveform fills the
envelope and the IQR instead of drawing hairlines, and a band says when columns
are sitting on the ceiling. The old pair drew the data correctly and gave you
no way to read it, on a tool whose subject is the two ends of the range.

THEME. theme.css is additive; style.css is untouched and the only markup change
is the titlebar. Five ids were added for the new controls and the smoke test
covers all five.

SCREENSHOT. docs/rudra_studio.png is the wipe -- baseline left, reconstruction
right, both at the same 1 000-nit display peak so the difference is the data
and not a grade. On that frame the SDR clips on 0.90% of pixels, the two sides
differ by a mean of 687 nits inside that region and by 0.0000 nits outside it.
"@

# ---- 2. the corpus work --------------------------------------------------
Commit-Group -Paths @(
    "docs/RETRAIN_RUNBOOK.md", "docs/TRAINING_STEPS.md",
    "training/fetch_corpus.py", "training/quarantine_broken.py"
) -Label "corpus" -Message @"
The next corpus and the next training run, both gated

docs/RETRAIN_RUNBOOK.md supersedes the order of work in docs/CORPUS.md: five
phases, each with a gate that stops it rather than a note to step over.

Phase 0 blocks the render. The corpus carries two luminance defects in opposite
directions from two ingest paths -- PolyHaven is relative data read as absolute
(p90/median 44.6, against 1.2 to 1.4 for every correctly scaled set) and the
Chimera path is a normalized 0..1 value that was never multiplied, so every
pair peaks below one nit. Together that is 19.5% of the corpus with a peak
luminance figure wrong by orders of magnitude.

Phase 3 carries the gate that decides whether the retrain is worth running at
all. Of 35 585 correctly decoded pairs not one contains a clipped pixel; all
205 clipped pairs are the broken PolyHaven decodes. If the 0 EV render still
does not clip, stop -- that answers the question for the price of a render.

docs/TRAINING_STEPS.md is phase 4 with the real flags, and its step 0 is a bug:
verify_dataset.py check 3 fails a corpus where too many records clip and has no
floor, so it passes ours at 0.57% affected. The gate cannot see the defect it
exists to catch.

training/quarantine_broken.py reports all three defects from pairs_index.jsonl.
training/fetch_corpus.py plans, checks, then fetches, recording licence,
licence_url and commercial_ok per set verbatim from the source; sets whose text
forbids training are listed and never fetched.
"@

# ---- 3. the cull ---------------------------------------------------------
if ($doomed) {
    git reset --quiet
    git rm --cached --quiet -- $doomed
    if ($LASTEXITCODE -ne 0) { Bail "git rm failed." }
    $msg = @"
Remove the working notes from the repo

.gitignore has had a blanket *.md rule for a while, with explicit negations for
the documents that are meant to be here. But .gitignore only governs what gets
ADDED: every note that went in before the rule was still tracked, and still on
the remote. This removes them from the index.

Kept: README.md, STATUS.md, HF_MODEL_CARD.md, checkpoints/README.md,
pipeline/README.md, docs/ and .github/ -- which is every markdown file that is
either linked from the README's documents table, read by a tool, or a package's
own README.

The content stays in history. Nothing here is unrecoverable, and the files stay
on disk locally; they are simply no longer part of what the repo publishes.

Removed:
$($doomed | ForEach-Object { "  $_" } | Out-String)
"@
    $f = Join-Path $tempRoot "rudra_msg_cull.txt"
    $msg | Set-Content -LiteralPath $f -Encoding ASCII
    git commit -F $f
    if ($LASTEXITCODE -ne 0) { Bail "git commit failed for the cull." }
    Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
    Write-Host "committed: cull ($($doomed.Count) files)" -ForegroundColor Green
}

# ---- 4. index clean, then push -------------------------------------------
git reset --quiet
$leftover = git --no-optional-locks status --porcelain --untracked-files=no
if ($leftover) {
    Write-Host ""
    Write-Host "Tracked files still differ from HEAD:" -ForegroundColor Yellow
    $leftover | ForEach-Object { Write-Host "   $_" }
    Bail "Refusing to push with an unclean tree. Deal with these first."
}

Write-Host ""
Write-Host "Fetching, then pushing." -ForegroundColor Cyan
git fetch origin
$behind = (git --no-optional-locks rev-list --count HEAD..origin/main).Trim()
if ($behind -ne "0") {
    Write-Host "origin is $behind commits ahead. Rebase first:" -ForegroundColor Red
    Write-Host "    powershell -ExecutionPolicy Bypass -File FINISH_REBASE.ps1" -ForegroundColor Cyan
    exit 1
}
git push
if ($LASTEXITCODE -ne 0) { Bail "Push failed. The commits stand; nothing is lost." }

Write-Host ""
Write-Host "Pushed." -ForegroundColor Green
git --no-optional-locks log --oneline -5
Write-Host ""
Write-Host "The UI smoke test needs the Studio running and is the check that" -ForegroundColor Yellow
Write-Host "matters for this push:" -ForegroundColor Yellow
Write-Host "    python ui\server.py" -ForegroundColor Yellow
Write-Host "    python tests\ui_smoke\press_everything.py" -ForegroundColor Yellow
