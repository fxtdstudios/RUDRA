# RUDRA -- clear the index, commit what the gitignore fix un-ignored, rebase, push.
#   powershell -ExecutionPolicy Bypass -File UNBLOCK_PUSH.ps1
#
# WHAT WENT WRONG
#
# FINISH_REBASE.ps1 refused with:
#
#     Uncommitted changes to tracked files. Commit or stash first:
#     A  HF_MODEL_CARD.md
#
# The porcelain code is "A " -- staged, added. Not a stray edit: a file sitting
# in the index that no commit picked up. Its guard is right to refuse, because a
# rebase replays commits over a staged addition and carries it into whichever
# one lands last.
#
# HF_MODEL_CARD.md was invisible to git for months under the blanket *.md rule
# -- the same rule that hid docs/RESULTS.md and cost us a05e383. The .gitignore
# commit in this stack negates it explicitly, which un-ignores the file. Nothing
# in COMMIT_ALL.ps1's five path lists claims it, and Commit-Group empties the
# index BEFORE each "git add" rather than after the last commit, so the file was
# staged by a force-add, never committed, and left there.
#
# So it is not a git problem and not a conflict. The file should be tracked --
# the HuggingFace upload reads it -- and the fix is to commit it, plus the
# one-line change to COMMIT_ALL.ps1 that stops this recurring.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

function Git-Run {
    param([string[]]$GitArgs)
    & git @GitArgs 2>&1 | ForEach-Object { Write-Host $_ }
    return $LASTEXITCODE
}
function Bail {
    param([string]$Why)
    Write-Host ""
    Write-Host $Why -ForegroundColor Red
    Write-Host "Nothing has been pushed. The working tree is untouched." -ForegroundColor Red
    exit 1
}

$before = (git --no-optional-locks rev-parse HEAD).Trim()
Write-Host "HEAD is $before" -ForegroundColor DarkGray
Write-Host "To undo every commit this script makes:  git reset --hard $before" -ForegroundColor DarkGray

Write-Host ""
Write-Host "Index before:" -ForegroundColor Cyan
$startState = git --no-optional-locks status --porcelain --untracked-files=no
if ($startState) { $startState | ForEach-Object { Write-Host "   $_" } } else { Write-Host "   (clean)" }

# ---- 1. empty the index --------------------------------------------------
# `git reset` with no --hard moves nothing on disk. Every file stays exactly as
# it is; only the staging area is cleared, so what follows stages deliberately
# instead of inheriting whatever an earlier script left behind.
git reset --quiet

# ---- 2. commit the files the gitignore fix un-ignored --------------------
# Restricted to the four paths .gitignore actually negates. A bare
# `ls-files --others` would also sweep up _conflict/ and any scratch file in the
# tree, which is how the first version of COMMIT_ALL put pilot_clipping.json
# into a commit called "Rewrite the paper as a paper".
$unignored = @(git --no-optional-locks ls-files --others --exclude-standard -- "HF_MODEL_CARD.md" "STATUS.md" "README.md" "docs/*.md")
if ($unignored) {
    Write-Host ""
    Write-Host "Un-ignored by the .gitignore fix, not yet tracked:" -ForegroundColor Cyan
    $unignored | ForEach-Object { Write-Host "   $_" }

    git add -f -- $unignored
    if ($LASTEXITCODE -ne 0) { Bail "git add failed." }

    $msgCards = @"
Track the model card the blanket *.md rule was hiding

HF_MODEL_CARD.md is read by the HuggingFace upload and has never been in the
repo, for the same reason docs/RESULTS.md was not: *.md in .gitignore with a
single negation for the root README. An ignored file is not an error, so the
absence reported itself nowhere.

The .gitignore commit earlier in this stack negates it explicitly. This commits
the file that negation exposes. It was staged by that run and left in the index
uncommitted, which is what stopped the rebase from starting.
"@
    $f = Join-Path ([System.IO.Path]::GetTempPath()) "rudra_msg_cards.txt"
    $msgCards | Set-Content -LiteralPath $f -Encoding ASCII
    git commit -F $f
    if ($LASTEXITCODE -ne 0) { Bail "git commit failed for the model card." }
    Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
    Write-Host "committed: model card" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Nothing un-ignored is untracked. Already committed." -ForegroundColor Yellow
}

# ---- 3. commit the script fix so this cannot recur ------------------------
git reset --quiet
$scriptDirty = git --no-optional-locks status --porcelain --untracked-files=no -- "COMMIT_ALL.ps1"
if ($scriptDirty) {
    git add -- "COMMIT_ALL.ps1"
    if ($LASTEXITCODE -ne 0) { Bail "git add failed for COMMIT_ALL.ps1." }

    $msgScript = @"
Leave the index clean when COMMIT_ALL finishes

Commit-Group empties the index before each "git add" and never after the last
commit, so anything staged outside the five path lists survives the whole run as
an addition nobody committed. That is how HF_MODEL_CARD.md ended up staged and
uncommitted, and the failure surfaced three scripts later as a rebase refusing
to start -- where it reads as a git problem rather than as this one.

A final "git reset" plus two reports: tracked files that still differ from HEAD,
and files the new .gitignore negations un-ignore that no path list claims.
"@
    $f = Join-Path ([System.IO.Path]::GetTempPath()) "rudra_msg_script.txt"
    $msgScript | Set-Content -LiteralPath $f -Encoding ASCII
    git commit -F $f
    if ($LASTEXITCODE -ne 0) { Bail "git commit failed for COMMIT_ALL.ps1." }
    Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
    Write-Host "committed: commit-script fix" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "COMMIT_ALL.ps1 matches HEAD already." -ForegroundColor Yellow
}

# ---- 4. the index has to be empty now, or the rebase will refuse again ----
git reset --quiet
$dirty = git --no-optional-locks status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host ""
    Write-Host "Tracked files still differ from HEAD:" -ForegroundColor Red
    $dirty | ForEach-Object { Write-Host "   $_" }
    Write-Host ""
    Write-Host "These are edits this script was not written to judge -- yours, or" -ForegroundColor Cyan
    Write-Host "another tool's. Commit them, or throw them away with git checkout, then re-run." -ForegroundColor Cyan
    exit 1
}
Write-Host ""
Write-Host "Index clean. The rebase can start." -ForegroundColor Green

# ---- 5. hand over to the rebase ------------------------------------------
Write-Host ""
Write-Host "Running FINISH_REBASE.ps1." -ForegroundColor Cyan
Write-Host ""
& powershell -ExecutionPolicy Bypass -File "FINISH_REBASE.ps1"
exit $LASTEXITCODE
