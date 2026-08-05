$ErrorActionPreference = "Continue"

$dlrmMain = "C:\Users\han\Desktop\dlrm\torchrec_dlrm\dlrm_main.py"
if ($env:DLRM_MAIN) {
    $dlrmMain = $env:DLRM_MAIN
}

Write-Host "DLRM profiler patch check"
Write-Host "========================="
Write-Host "Target: $dlrmMain"

if (-not (Test-Path $dlrmMain)) {
    Write-Host "[MISSING] dlrm_main.py was not found."
    exit 1
}

$content = Get-Content -LiteralPath $dlrmMain -Raw
$required = @(
    "--profile_dir",
    "--profile_record_shapes",
    "--profile_memory",
    "_maybe_profile",
    "torch.profiler.profile",
    "export_chrome_trace"
)

$ok = $true
foreach ($item in $required) {
    if ($content.Contains($item)) {
        Write-Host "[OK] $item"
    } else {
        Write-Host "[MISSING] $item"
        $ok = $false
    }
}

if ($ok) {
    Write-Host "DLRM profiler patch is present."
    exit 0
}

Write-Host "DLRM profiler patch is incomplete."
exit 1
