#Requires -Version 5.1
<#
    One-click launcher for the web version.

    Creates a .venv if needed, installs Python + Node dependencies, starts the
    FastAPI backend and the Next.js dev server, then opens the browser.

    Usage:
        .\run_web.ps1              # CPU torch (fast, works everywhere)
        .\run_web.ps1 -Gpu         # CUDA 12.6 torch (large download)
        .\run_web.ps1 -Reinstall   # force dependency reinstall
        .\run_web.ps1 -SkipInstall # never touch pip/npm
#>
[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Reinstall,
    [switch]$SkipInstall,
    [int]$ApiPort = 8000,
    [int]$WebPort = 3000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

function Info($m) { Write-Host "[web] $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[web] ERROR: $m" -ForegroundColor Red; exit 1 }

function Stop-Tree($p) {
    if ($null -eq $p) { return }
    try { & taskkill /PID $p.Id /T /F 2>&1 | Out-Null } catch { }
}

function Wait-Url($url, $seconds, $proc) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $proc -and $proc.HasExited) { return $false }
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { return $true }
        } catch { }
        Start-Sleep -Milliseconds 750
    }
    return $false
}

# --- prerequisites ---------------------------------------------------------
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail "Node.js not found. Install it from https://nodejs.org and re-run."
}

$pyExe = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) { $pyExe = 'py'; $pyArgs = @('-3') }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $pyExe = 'python' }
else { Fail "Python 3 not found. Install it from https://python.org and re-run." }

# --- python venv -----------------------------------------------------------
$venvPy = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) {
    Info "Creating virtual environment (.venv)..."
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create the virtual environment." }
}

$flavor = if ($Gpu) { 'gpu' } else { 'cpu' }
$marker = Join-Path $Root ".venv\.idbu-ready-$flavor"
if ((-not (Test-Path -LiteralPath $marker)) -or $Reinstall) {
    if ($SkipInstall) {
        Info "Skipping dependency install (-SkipInstall)."
    }
    else {
        Info "Installing Python dependencies (first run can take a few minutes)..."
        & $venvPy -m pip install --upgrade pip
        if ($Gpu) {
            & $venvPy -m pip install -r requirements.txt
        }
        else {
            $cpuReq = Join-Path $env:TEMP 'idbu-requirements-cpu.txt'
            Get-Content requirements.txt |
                Where-Object { $_ -notmatch 'torch' -and $_ -notmatch 'extra-index-url' } |
                Set-Content -LiteralPath $cpuReq -Encoding ASCII
            & $venvPy -m pip install -r $cpuReq
            if ($LASTEXITCODE -ne 0) { Fail "Python dependency install failed." }
            Info "Installing CPU torch (use -Gpu for CUDA)..."
            & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        }
        if ($LASTEXITCODE -ne 0) { Fail "Python dependency install failed." }
        New-Item -ItemType File -Path $marker -Force | Out-Null
    }
}

# --- node deps -------------------------------------------------------------
$nodeModules = Join-Path $Root 'node_modules'
if ((-not (Test-Path -LiteralPath $nodeModules)) -or $Reinstall) {
    if ($SkipInstall) { Fail "node_modules is missing and -SkipInstall was set." }
    Info "Installing Node dependencies..."
    & cmd.exe /c npm install
    if ($LASTEXITCODE -ne 0) { Fail "npm install failed." }
}

# --- launch ----------------------------------------------------------------
$logDir = Join-Path $Root '.tmp'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$env:NEXT_PUBLIC_API_BASE = "http://127.0.0.1:$ApiPort"

Info "Starting API on http://127.0.0.1:$ApiPort ..."
$api = Start-Process -FilePath $venvPy `
    -ArgumentList @('-m', 'uvicorn', 'api_backend.app:app', '--host', '127.0.0.1', '--port', "$ApiPort") `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir 'api.out.log') `
    -RedirectStandardError (Join-Path $logDir 'api.err.log')

if (-not (Wait-Url "http://127.0.0.1:$ApiPort/health" 180 $api)) {
    Stop-Tree $api
    Fail "API did not become ready. See .tmp\api.err.log"
}

Info "Starting web UI on http://localhost:$WebPort ..."
$web = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/c', 'npm', 'run', 'dev', '--', '-p', "$WebPort") `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir 'web.out.log') `
    -RedirectStandardError (Join-Path $logDir 'web.err.log')

if (-not (Wait-Url "http://127.0.0.1:$WebPort" 180 $web)) {
    Stop-Tree $web
    Stop-Tree $api
    Fail "Web UI did not become ready. See .tmp\web.err.log"
}

if (-not $NoBrowser) { Start-Process "http://localhost:$WebPort" }
Info "Ready at http://localhost:$WebPort  (API: http://127.0.0.1:$ApiPort)"
Info "Press Ctrl+C to stop."

try {
    Wait-Process -Id $web.Id
}
finally {
    Info "Stopping..."
    Stop-Tree $api
    Stop-Tree $web
}
