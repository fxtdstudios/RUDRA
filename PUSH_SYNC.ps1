# RUDRA -- look at the divergence before touching it, then sync and push.
#   powershell -ExecutionPolicy Bypass -File PUSH_SYNC.ps1
#   powershell -ExecutionPolicy Bypass -File PUSH_SYNC.ps1 -Rebase
#
# The push was rejected because origin has commits this clone does not. Nothing
# is lost: your delivery commit is here, committed, just not pushed. What is
# not safe is reacting blindly. A plain "git pull" writes a merge commit into
# the history; a force push throws away whatever is on the remote. So this
# looks first, prints both sides, and only rewrites anything when you ask.
#
# Run it with no arguments and read the two lists. If the remote side is
# something you recognise, run it again with -Rebase: your commits are replayed
# on top of the remote, the tests run again on the combined tree, and only then
# does it push. No force push, ever, from this script.
param([switch]$Rebase)

# Deliberately NOT "Stop": git writes ordinary progress to stderr, and with
# Stop set that becomes a terminating error in PowerShell 5.1 and kills the
# script halfway through. Every git call below is checked by its exit code.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

function Git-Run {
    param([string[]]$GitArgs)
    & git @GitArgs 2>&1 | ForEach-Object { Write-Host $_ }
    return $LASTEXITCODE
}

function Git-Read {
    param([string[]]$GitArgs)
    $out = & git --no-optional-locks @GitArgs 2>$null
    return @{ code = $LASTEXITCODE; text = ($out | Out-String) }
}

$head = Git-Read @("rev-parse", "--abbrev-ref", "HEAD")
if ($head.code -ne 0) { Write-Host "not a git repo here" -ForegroundColor Red; exit 1 }
$branch = $head.text.Trim()
if ($branch -eq "HEAD") {
    Write-Host "HEAD is detached, not on a branch. Stopping." -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "branch: $branch" -ForegroundColor Cyan

if ((Git-Run @("fetch", "origin")) -ne 0) { Write-Host "fetch failed" -ForegroundColor Red; exit 1 }

$probe = Git-Read @("rev-parse", "--verify", "origin/$branch")
if ($probe.code -ne 0) {
    Write-Host "origin/$branch does not exist. Stopping." -ForegroundColor Red
    exit 1
}

$counts = Git-Read @("rev-list", "--left-right", "--count", "origin/$branch...HEAD")
if ($counts.code -ne 0) { Write-Host "could not compare with origin" -ForegroundColor Red; exit 1 }
$parts = $counts.text.Trim() -split "\s+" | Where-Object { $_ -ne "" }
$behind = [int]$parts[0]
$ahead  = [int]$parts[1]

Write-Host ""
Write-Host "$ahead commit(s) here that origin does not have; $behind on origin that you do not." -ForegroundColor Cyan

if ($behind -gt 0) {
    Write-Host ""
    Write-Host "ON THE REMOTE, NOT HERE ($behind):" -ForegroundColor Yellow
    & git --no-optional-locks log --oneline --no-decorate "HEAD..origin/$branch"
    Write-Host ""
    Write-Host "what those commits touch:" -ForegroundColor Yellow
    & git --no-optional-locks diff --stat "HEAD...origin/$branch"
}
if ($ahead -gt 0) {
    Write-Host ""
    Write-Host "HERE, NOT ON THE REMOTE ($ahead):" -ForegroundColor Green
    & git --no-optional-locks log --oneline --no-decorate "origin/$branch..HEAD"
}

if ($behind -eq 0) {
    if ($ahead -eq 0) {
        Write-Host ""
        Write-Host "Already in sync, nothing to do." -ForegroundColor Green
        exit 0
    }
    Write-Host ""
    Write-Host "Remote had nothing new after the fetch. Pushing." -ForegroundColor Cyan
    if ((Git-Run @("push")) -ne 0) { Write-Host "push failed" -ForegroundColor Red; exit 1 }
    Write-Host "Pushed." -ForegroundColor Green
    exit 0
}

if (-not $Rebase) {
    Write-Host ""
    Write-Host "Stopping here on purpose. Read the remote list above." -ForegroundColor Cyan
    Write-Host "If you recognise it, run:" -ForegroundColor Cyan
    Write-Host "   powershell -ExecutionPolicy Bypass -File PUSH_SYNC.ps1 -Rebase"
    Write-Host "If you do not recognise it, send me the output and stop." -ForegroundColor Cyan
    exit 0
}

# -- -Rebase from here ----------------------------------------------------
# --untracked-files=no on purpose: a rebase cannot touch a file git is not
# tracking, and this repo always has untracked outputs, checkpoints and
# __pycache__ lying around. Uncommitted changes to TRACKED files are the real
# hazard, because a rebase carries them across commits and they get lost.
$status = Git-Read @("status", "--porcelain", "--untracked-files=no")
$dirty = $status.text.Trim()
if ($dirty) {
    Write-Host ""
    Write-Host "You have uncommitted changes to tracked files. A rebase carries them" -ForegroundColor Red
    Write-Host "across and that is how edits get lost, so this stops. Commit or stash" -ForegroundColor Red
    Write-Host "these first (untracked files are fine and were ignored here):" -ForegroundColor Red
    Write-Host $dirty
    exit 1
}

$before = (Git-Read @("rev-parse", "HEAD")).text.Trim()
Write-Host ""
Write-Host "Replaying $ahead commit(s) onto origin/$branch." -ForegroundColor Cyan
Write-Host "Pre-rebase HEAD is $before" -ForegroundColor DarkGray
Write-Host "To undo the rebase completely at any point:  git reset --hard $before" -ForegroundColor DarkGray

if ((Git-Run @("pull", "--rebase", "origin", $branch)) -ne 0) {
    Write-Host ""
    Write-Host "The rebase did not apply cleanly." -ForegroundColor Yellow

    # Write the three sides of every conflict out BEFORE aborting. The index
    # stages only exist while the rebase is in progress, and "paste me the
    # conflict" has cost several round trips on this repo already. With the
    # three files on disk a merge can be done properly, by hand or by me,
    # instead of guessed at from a diff someone retyped.
    $dump = Join-Path (Get-Location) "_conflict"
    New-Item -ItemType Directory -Force -Path $dump | Out-Null
    # Kept out of git without touching .gitignore, which is a tracked file and
    # not the place for one machine's scratch directory.
    $exclude = ".git/info/exclude"
    if ((Test-Path -LiteralPath $exclude) -and
        -not (Select-String -Path $exclude -Pattern '^_conflict/$' -Quiet)) {
        Add-Content -LiteralPath $exclude -Value "_conflict/"
    }
    $conflicted = git --no-optional-locks diff --name-only --diff-filter=U
    foreach ($f in $conflicted) {
        $safe = ($f -replace '[\\/]', '_')
        foreach ($stage in @(@(1, "base"), @(2, "theirs"), @(3, "ours"))) {
            $out = Join-Path $dump ("{0}.{1}" -f $safe, $stage[1])
            & git --no-optional-locks show (":{0}:{1}" -f $stage[0], $f) 2>$null |
                Set-Content -LiteralPath $out -Encoding UTF8
        }
        Write-Host "  conflict in $f" -ForegroundColor Yellow
    }
    # During a rebase "ours" is the upstream you are landing on and "theirs" is
    # the commit being replayed, which is the opposite of what the words mean
    # everywhere else. The names above are already corrected for that: .theirs
    # is what is on origin, .ours is your commit.
    Write-Host ""
    Write-Host "Wrote the three sides of each conflict to:" -ForegroundColor Cyan
    Write-Host "    $dump" -ForegroundColor Cyan
    Write-Host "  <file>.base    the common ancestor" -ForegroundColor DarkGray
    Write-Host "  <file>.theirs  what is on origin" -ForegroundColor DarkGray
    Write-Host "  <file>.ours    your commit" -ForegroundColor DarkGray

    Git-Run @("rebase", "--abort") | Out-Null
    Write-Host ""
    Write-Host "Rebase backed out. The repo is exactly as it was and nothing is lost." -ForegroundColor Green
    Write-Host "Send me the _conflict folder, or merge it yourself and re-run with -Rebase." -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "Tests, on the combined tree. Someone else's commits are in it now." -ForegroundColor Cyan
$tempRoot = $env:TEMP
if (-not $tempRoot) { $tempRoot = [System.IO.Path]::GetTempPath() }
$basetemp = Join-Path $tempRoot "rudra-pytest"
python -m pytest tests/ -q --basetemp="$basetemp"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Tests fail after the rebase. NOT pushing." -ForegroundColor Red
    Write-Host "To undo the rebase entirely:  git reset --hard $before" -ForegroundColor Red
    exit 1
}

if ((Git-Run @("push")) -ne 0) { Write-Host "push failed" -ForegroundColor Red; exit 1 }
Write-Host ""
Write-Host "Rebased, tested and pushed." -ForegroundColor Green
& git --no-optional-locks log --oneline -3
