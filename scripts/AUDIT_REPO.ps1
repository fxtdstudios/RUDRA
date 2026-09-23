# RUDRA -- security and provenance audit.
#
# Answers three questions over the WORKING TREE and the FULL GIT HISTORY,
# because a secret deleted in a later commit is still in the history and still
# published:
#
#   1. Are there credentials in here -- ours or anyone's?
#   2. What is tracked that should not be?
#   3. Where does the repo still carry AI-assistant attribution?
#
# Read-only. It changes nothing, stages nothing and commits nothing. Every git
# call uses --no-optional-locks so it cannot leave a .git\index.lock behind.
#
#   powershell -ExecutionPolicy Bypass -File AUDIT_REPO.ps1

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

function Write-Head {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor DarkGray
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor DarkGray
}

function Write-Clean { Write-Host "  none" -ForegroundColor Green }

# Provider key SHAPES. Deliberately shape-based: grepping for the word "key"
# drowns in false positives and still misses the one line that matters.
$secretPattern = 'sk-ant-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|glpat-[A-Za-z0-9_-]{20,}|BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY'

# Assignments with a literal on the right. Reading from the environment is the
# correct pattern, so those lines are filtered out below.
$assignPattern = '(api[_-]?key|apikey|secret|passwd|password|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)[[:space:]]*[:=][[:space:]]*.{8,}'
$benign = 'os\.environ|getenv|process\.env|placeholder|example|your_|dummy|redacted|\$env:|argparse|help='

Write-Head "1a. Credential shapes -- WORKING TREE (tracked files)"
$hits = @(git --no-optional-locks grep -n -I -E $secretPattern -- . 2>$null)
if ($hits.Count -gt 0) { $hits | ForEach-Object { Write-Host "  $_" -ForegroundColor Red } } else { Write-Clean }

Write-Head "1b. Credential shapes -- FULL HISTORY (every commit ever)"
$revs = @(git --no-optional-locks rev-list --all 2>$null)
Write-Host ("  {0} commits to scan..." -f $revs.Count)
$histHits = New-Object System.Collections.ArrayList
# Chunked: Windows caps a command line near 32k characters, and one rev is 41
# of them, so a long history passed in one call fails with a truncated arg list
# rather than an honest error.
for ($i = 0; $i -lt $revs.Count; $i += 150) {
    $end = [Math]::Min($i + 149, $revs.Count - 1)
    $chunk = $revs[$i..$end]
    $found = @(git --no-optional-locks grep -n -I -E $secretPattern $chunk -- . 2>$null)
    foreach ($line in $found) { [void]$histHits.Add($line) }
}
if ($histHits.Count -gt 0) {
    $histHits | Select-Object -Unique | Select-Object -First 40 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  A secret in history is still published. ROTATE THE KEY FIRST." -ForegroundColor Yellow
    Write-Host "  Rewriting history does not un-leak what was already fetched." -ForegroundColor Yellow
} else { Write-Clean }

Write-Head "1c. Hard-coded secret assignments -- working tree"
$assign = @(git --no-optional-locks grep -n -I -E $assignPattern -- '*.py' '*.js' '*.json' '*.yml' '*.yaml' '*.toml' '*.ps1' '*.bat' '*.sh' 2>$null)
$assign = @($assign | Where-Object { $_ -notmatch $benign })
if ($assign.Count -gt 0) { $assign | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow } } else { Write-Clean }

Write-Head "2a. Secret-bearing files that are TRACKED"
$tracked = @(git --no-optional-locks ls-files 2>$null)
$risky = @($tracked | Where-Object {
    $_ -match '(^|/)\.env' -or
    $_ -match '\.(pem|key|pfx|p12|keystore|jks|ppk)$' -or
    $_ -match '(^|/)(id_rsa|id_ed25519|credentials|secrets?)(\.|$)'
})
if ($risky.Count -gt 0) { $risky | ForEach-Object { Write-Host "  TRACKED: $_" -ForegroundColor Red } } else { Write-Clean }

Write-Head "2b. Local helper scripts that are TRACKED"
$helpers = @($tracked | Where-Object {
    $_ -match '^(COMMIT|AUDIT|RUN)_.*\.ps1$' -or $_ -eq 'AGENTS.md' -or $_ -eq 'COMMITMSG.txt'
})
if ($helpers.Count -gt 0) {
    $helpers | ForEach-Object { Write-Host "  TRACKED: $_" -ForegroundColor Yellow }
    Write-Host "  untrack without deleting:  git rm --cached <file>" -ForegroundColor DarkGray
} else { Write-Host "  none -- all local-only" -ForegroundColor Green }

Write-Head "3a. Assistant attribution -- WORKING TREE (tracked files)"
# The bracketed first letters keep this script from matching its own patterns.
$attr = @(git --no-optional-locks grep -n -i -I -E '[c]laude|[a]nthropic|co-authored-by|generated with' -- . 2>$null)
if ($attr.Count -gt 0) { $attr | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow } } else { Write-Clean }

Write-Head "3b. Assistant attribution -- COMMIT MESSAGES"
$msgs = @(git --no-optional-locks log --all --oneline -i --grep='[c]laude' --grep='[a]nthropic' --grep='co-authored-by' 2>$null)
if ($msgs.Count -gt 0) {
    Write-Host ("  {0} commit(s) mention it. First 15:" -f $msgs.Count) -ForegroundColor Yellow
    $msgs | Select-Object -First 15 | ForEach-Object { Write-Host "    $_" }
    Write-Host ""
    Write-Host "  Removing these REWRITES HISTORY: every hash from the earliest" -ForegroundColor Yellow
    Write-Host "  one onward changes and the push must be forced. Do it only if" -ForegroundColor Yellow
    Write-Host "  nobody else has this repo cloned. Three steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    git branch backup-before-rewrite" -ForegroundColor DarkGray
    Write-Host "    pip install git-filter-repo" -ForegroundColor DarkGray
    Write-Host "    git filter-repo --replace-message SCRUB_TRAILERS.txt" -ForegroundColor DarkGray
    Write-Host "    git push --force-with-lease --all" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  filter-repo removes the remote afterwards by design; re-add it" -ForegroundColor DarkGray
    Write-Host "  with: git remote add origin <url>   (3c prints the url below)" -ForegroundColor DarkGray
} else { Write-Clean }

Write-Head "3c. Commit AUTHORS and COMMITTERS"
$people = @(git --no-optional-locks log --all --format='%an <%ae>' 2>$null) +
          @(git --no-optional-locks log --all --format='%cn <%ce>' 2>$null)
$people | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }

Write-Head "Remote"
git --no-optional-locks remote -v

Write-Host ""
Write-Host "Audit complete. Nothing was changed." -ForegroundColor Green
