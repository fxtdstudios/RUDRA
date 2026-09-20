# RUDRA -- strip the byte order mark from the tip commit's message.
#
# Set-Content -Encoding UTF8 on PowerShell 5.1 writes a BOM. git kept it, so
# the subject of 89f9ae7 begins with U+FEFF and renders on GitHub with a stray
# character before "ui:". Nothing else about the commit is wrong.
#
# This rewrites ONE commit -- the tip -- and force-pushes it. The tree, the
# author, the date and the message text are unchanged; only the leading BOM
# goes, so the commit hash changes and nothing else does. Safe on a repo only
# you have cloned. If someone else has it cloned, they will need to reset.
#
#   powershell -ExecutionPolicy Bypass -File FIX_COMMIT_BOM.ps1

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

$bom = [char]0xFEFF
$subject = (git --no-optional-locks log -1 --format=%s)
if (-not $subject.StartsWith($bom)) {
    Write-Host "No BOM on the tip commit -- nothing to do." -ForegroundColor Green
    git --no-optional-locks log --oneline -1
    exit 0
}

Write-Host "Tip commit subject starts with a BOM. Rewriting it." -ForegroundColor Yellow
git --no-optional-locks log --oneline -1

# %B is the raw body. Join the lines back with newlines, drop the BOM, and
# write the file with an explicit no-BOM encoder rather than trusting a
# -Encoding name -- that is what produced the problem in the first place.
$lines = @(git --no-optional-locks log -1 --format=%B)
$message = ($lines -join "`n").TrimStart($bom).TrimEnd("`n") + "`n"

$tempRoot = $env:TEMP
if (-not $tempRoot) { $tempRoot = [System.IO.Path]::GetTempPath() }
$msgFile = Join-Path $tempRoot "rudra_amend_msg.txt"
[System.IO.File]::WriteAllText($msgFile, $message, (New-Object System.Text.UTF8Encoding($false)))

git commit --amend -F $msgFile
if ($LASTEXITCODE -ne 0) { Write-Host "amend failed" -ForegroundColor Red; exit 1 }
Remove-Item -LiteralPath $msgFile -Force -ErrorAction SilentlyContinue

# --force-with-lease refuses if the remote moved since your last fetch, so a
# push someone else made cannot be silently discarded. Plain --force would.
git push --force-with-lease
if ($LASTEXITCODE -ne 0) { Write-Host "push failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Done." -ForegroundColor Green
git --no-optional-locks log --oneline -1
