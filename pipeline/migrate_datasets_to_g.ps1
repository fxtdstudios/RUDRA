# RUDRA Dataset Migration Worker
# Automatically finishes Stuttgart, migrates Netflix Chimera and HdM, sets up junction.
$ErrorActionPreference = "Continue"

Write-Host "=== RUDRA Dataset Migration to G:\datasets\sources ===" -ForegroundColor Cyan

# 1. Wait for Stuttgart transfer to complete if currently running
Write-Host "Waiting for Stuttgart_HDR_2014 transfer to finish..." -ForegroundColor Yellow
while ($true) {
    $left = (Get-ChildItem -Path "E:\source_hdr\Stuttgart_HDR_2014" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($left -eq 0) {
        Write-Host "Stuttgart transfer complete!" -ForegroundColor Green
        break
    }
    $cnt = (Get-ChildItem -Path "G:\datasets\sources\stuttgart_hdr_2014" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "  Stuttgart in progress... $cnt files on G: ($left remaining on E:)" -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
}

# 2. Transfer Netflix Chimera
if (Test-Path "E:\source_hdr\Netflix") {
    Write-Host "`nStarting Netflix Chimera transfer (250 GB)..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "G:\datasets\sources\netflix_chimera" -Force | Out-Null
    $srcNet = if (Test-Path "E:\source_hdr\Netflix\tif_DCI4k2398p") { "E:\source_hdr\Netflix\tif_DCI4k2398p" } else { "E:\source_hdr\Netflix" }
    robocopy $srcNet "G:\datasets\sources\netflix_chimera" /MOVE /E /MT:16 /J /R:3 /W:2 /NP /NFL /NDL /NJH /NJS
    Remove-Item "E:\source_hdr\Netflix" -Recurse -Force -ErrorAction SilentlyContinue
    $c = (Get-ChildItem -Path "G:\datasets\sources\netflix_chimera" -Recurse -File | Measure-Object).Count
    Write-Host "Netflix Chimera transfer complete! Total files: $c" -ForegroundColor Green
}

# 3. Transfer HdM-HFR-2017
if (Test-Path "E:\source_hdr\HdM-HFR-2017_Color-Graded") {
    Write-Host "`nStarting HdM-HFR-2017 transfer (290 GB)..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "G:\datasets\sources\hdm_hfr_2017" -Force | Out-Null
    robocopy "E:\source_hdr\HdM-HFR-2017_Color-Graded" "G:\datasets\sources\hdm_hfr_2017" /MOVE /E /MT:16 /J /R:3 /W:2 /NP /NFL /NDL /NJH /NJS
    Remove-Item "E:\source_hdr\HdM-HFR-2017_Color-Graded" -Recurse -Force -ErrorAction SilentlyContinue
    $c = (Get-ChildItem -Path "G:\datasets\sources\hdm_hfr_2017" -Recurse -File | Measure-Object).Count
    Write-Host "HdM-HFR-2017 transfer complete! Total files: $c" -ForegroundColor Green
}

# 4. Remove empty E:\source_hdr and create backward-compatibility Junction
if (Test-Path "E:\source_hdr") {
    $remaining = Get-ChildItem "E:\source_hdr"
    if ($remaining.Count -eq 0) {
        Remove-Item "E:\source_hdr" -Force -Recurse
        Write-Host "Creating NTFS junction E:\source_hdr -> G:\datasets\sources..." -ForegroundColor Cyan
        New-Item -ItemType Junction -Path "E:\source_hdr" -Target "G:\datasets\sources" | Out-Null
        Write-Host "Junction created successfully!" -ForegroundColor Green
    } else {
        Write-Host "Warning: E:\source_hdr still has unhandled files: $($remaining.Name -join ', ')" -ForegroundColor Red
    }
}

# 5. Final audit
Write-Host "`n=== FINAL AUDIT OF G:\datasets\sources ===" -ForegroundColor Cyan
Get-ChildItem -Path "G:\datasets\sources" -Directory | ForEach-Object {
    $files = Get-ChildItem -LiteralPath $_.FullName -Recurse -File -ErrorAction SilentlyContinue
    $count = ($files | Measure-Object).Count
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Dataset = $_.Name
        Files = $count
        SizeGB = [math]::Round($bytes / 1GB, 2)
    }
} | Format-Table -AutoSize

Get-PSDrive G, E | Select-Object Name, @{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}, @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}} | Format-Table -AutoSize
Write-Host "Migration workflow finished successfully!" -ForegroundColor Green
