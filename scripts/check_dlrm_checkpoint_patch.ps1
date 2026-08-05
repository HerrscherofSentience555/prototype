$ErrorActionPreference = "Continue"

$dlrmMain = "C:\Users\han\Desktop\dlrm\torchrec_dlrm\dlrm_main.py"
if ($env:DLRM_MAIN) {
    $dlrmMain = $env:DLRM_MAIN
}

Write-Host "DLRM checkpoint patch check"
Write-Host "==========================="
Write-Host "Target: $dlrmMain"

if (-not (Test-Path $dlrmMain)) {
    Write-Host "[MISSING] dlrm_main.py was not found."
    exit 1
}

$content = Get-Content -LiteralPath $dlrmMain -Raw
$required = @(
    "--checkpoint_save_dir",
    "--checkpoint_load_path",
    "--checkpoint_save_optimizer",
    "_load_checkpoint_if_needed",
    "_save_checkpoint_if_needed",
    "weights_only=False"
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
    Write-Host "DLRM checkpoint patch is present."
    exit 0
}

Write-Host "DLRM checkpoint patch is incomplete."
exit 1
