$ErrorActionPreference = "Continue"

Write-Host "TorchRec Prototype Windows Environment Check"
Write-Host "==========================================="

function Test-CommandAvailable {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        Write-Host "[OK] $Name -> $($command.Source)"
        return $true
    }
    Write-Host "[MISSING] $Name"
    return $false
}

$ok = $true
$ok = (Test-CommandAvailable "python") -and $ok
$ok = (Test-CommandAvailable "wsl") -and $ok

if (Test-Path ".\requirements.txt") {
    Write-Host "[OK] requirements.txt found"
} else {
    Write-Host "[MISSING] requirements.txt"
    $ok = $false
}

python -c "import gradio, pandas, pydantic, yaml; print('[OK] core Python packages import')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[MISSING] one or more core Python packages; run: pip install -r requirements.txt"
    $ok = $false
}

if ($ok) {
    Write-Host "Environment check passed."
    exit 0
}

Write-Host "Environment check failed."
exit 1
