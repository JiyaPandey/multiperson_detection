$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Get-PythonVersion {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return (& py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    }

    return $null
}

function Test-PythonVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VersionString
    )

    $parts = $VersionString.Split('.')
    if ($parts.Length -lt 2) {
        return $false
    }

    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 10)
}

Write-Host "[setup] Checking Python installation..." -ForegroundColor Cyan
$pythonVersion = Get-PythonVersion

if (-not $pythonVersion) {
    Write-Host "[setup] Python not found. Install Python 3.10+ and re-run .\\setup.ps1" -ForegroundColor Red
    exit 1
}

if (-not (Test-PythonVersion -VersionString $pythonVersion)) {
    Write-Host "[setup] Found Python $pythonVersion. Python 3.10+ is required." -ForegroundColor Red
    exit 1
}

Write-Host "[setup] Python $pythonVersion detected." -ForegroundColor Green

$venvPython = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[setup] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    }
    else {
        & python -m venv .venv
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Host "[setup] Failed to create .venv." -ForegroundColor Red
    exit 1
}

Write-Host "[setup] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "" 
Write-Host "[setup] Done. Run the dashboard with:" -ForegroundColor Green
Write-Host ".\\.venv\\Scripts\\python.exe -m streamlit run src\\ui\\dashboard.py" -ForegroundColor Yellow
