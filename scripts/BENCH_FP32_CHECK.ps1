# RUDRA -- is the bench gap bf16? Re-export four CP7 rows in fp32 and compare.
#   pwsh -File D:\A.I\Devlopments\rudra\scripts\BENCH_FP32_CHECK.ps1
#
# Why: training evals (fp32) put v4b at +1.4 dB clean over the analytic
# inverse; CP7 (bf16 autocast in predict_image) put it at -8.1 dB PU21 on the
# same ACES render, median frame 38.1 dB against the baseline's 47.3. The
# baseline tree is fp32 and bf16 keeps 8 mantissa bits, so part of that gap
# may be the harness, not the model. This answers it in about 40 minutes.
#
# Writes <model>_fp32 trees beside the bf16 ones in bench\cp_aces / cp_oog /
# cp_mix (ref/ and baseline/ are reused), scores them, then re-runs
# training\cp7_verdicts.py. Logs: reports\logs\fp32_*.log. Resumable: a tree
# already scored is skipped. Nothing is committed.
$ErrorActionPreference = "Continue"
$Repo = "D:\A.I\Devlopments\rudra"
Set-Location -LiteralPath $Repo
$logDir = Join-Path $Repo "reports\logs"; New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$py = if (Test-Path ".venv\Scripts\python.exe") { (Resolve-Path ".venv\Scripts\python.exe").Path } else { "python" }
$env:OPENCV_IO_ENABLE_OPENEXR = "1"
function Run([string[]]$argv, [string]$log) {
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir $log) | Out-Host
    return $LASTEXITCODE
}
function Fail([string]$why) { Write-Host "`nSTOPPED: $why" -ForegroundColor Red; exit 1 }

$v4b = @("G:\corpus_v4b", "G:\datasets\corpora\corpus_v4b") | Where-Object { Test-Path (Join-Path $_ "sdr_hdr_manifest.jsonl") } | Select-Object -First 1
$v4c = "G:\datasets\corpora\corpus_v4c"
$holdOut = "E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl"
if (-not $v4b) { Fail "corpus_v4b manifest not found" }
$ckpt = @{ "v4b" = "checkpoints\sdr2hdr_image_v4\best.pt"; "v4c" = "checkpoints\sdr2hdr_image_v4c\best.pt" }
# bench, model, manifest, condition -- the same arguments CP7 used, plus --precision fp32
$rows = @(
    @("aces", "v4b", (Join-Path $v4b "sdr_hdr_manifest.jsonl"), "clean"),
    @("aces", "v4c", (Join-Path $v4b "sdr_hdr_manifest.jsonl"), "clean"),
    @("oog",  "v4c", $holdOut, "out-of-generator"),
    @("mix",  "v4c", (Join-Path $v4c "sdr_hdr_manifest.jsonl"), "clean")
)
foreach ($r in $rows) {
    $b, $m, $manifest, $cond = $r
    $root = Join-Path $Repo "bench\cp_$b"; $res = Join-Path $root "results"; $tree = "${m}_fp32"
    if (-not (Test-Path (Join-Path $root "ref"))) { Fail "bench\cp_$b has no ref\ tree; run CP7 first" }
    if (-not (Test-Path (Join-Path $res "$tree.csv"))) {
        Write-Host "`n== $b / $m  fp32" -ForegroundColor Cyan
        if (-not (Test-Path (Join-Path $root $tree))) {
            $a = @("training\export_bench_pairs.py", "--checkpoint", $ckpt[$m], "--manifest", $manifest, "--out", $root,
                   "--split", "test", "--condition", $cond, "--test-name", $tree, "--only-test", "--precision", "fp32")
            if ((Run $a "fp32_export_${b}_$m.log") -ne 0) { Fail "export $b/$m failed -- reports\logs\fp32_export_${b}_$m.log" }
        }
        if ((Run @("-m", "rudra.delivery.cli", "bench", $root, "--nits-scale", "203", "--test-dir", $tree, "--output", (Join-Path $res "$tree.json")) "fp32_bench_${b}_$m.log") -ne 0) { Fail "bench $b/$tree failed" }
    }
    Write-Host "`n-- $b/$m : fp32 against bf16 (report only)" -ForegroundColor Yellow
    Run @("training\paired_gate.py", "--a", (Join-Path $res "$tree.csv"), "--b", (Join-Path $res "$m.csv"), "--out", (Join-Path $res "fp32_${m}_vs_bf16.json")) "fp32_vs_bf16_${b}_$m.log" | Out-Null
}
Write-Host "`n== verdicts (the *_fp32 rows are the new ones)" -ForegroundColor Cyan
Run @("training\cp7_verdicts.py") "fp32_verdicts.log" | Out-Null
Write-Host @"

Reading it:
  fp32 vs bf16 near zero        -> precision is not the gap. The model really is worse
                                   than the inverse on the bench; the next thing to fix
                                   is best.pt selection (32 val crops, log1p metric).
  aces/v4b_fp32 vs baseline >> -8 dB
                                   -> the bench was penalising bf16. Re-export every CP7
                                   row with --precision fp32 before reading any gate,
                                   and make fp32 the export default.
"@ -ForegroundColor Green
