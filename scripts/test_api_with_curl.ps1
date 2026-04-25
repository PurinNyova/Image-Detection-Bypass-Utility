[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [string]$PythonExe = '',
    [string[]]$SkipStages = @(),
    [switch]$KeepArtifacts
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$SkipStages = @(
    $SkipStages |
        ForEach-Object { $_ -split ',' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)

$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$TempRoot = Join-Path $WorkspaceRoot '.tmp\curl-api-tests'
$ArtifactsDir = Join-Path $TempRoot 'artifacts'
$FixturesDir = Join-Path $TempRoot 'fixtures'
$HeadersDir = Join-Path $TempRoot 'headers'
$ConfigsDir = Join-Path $TempRoot 'configs'

$serverProcess = $null

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Resolve-PythonExecutable {
    if ($PythonExe) {
        return $PythonExe
    }

    $candidates = @(
        (Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'),
        (Join-Path $WorkspaceRoot '.venv\bin\python'),
        'python'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }

        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
    }

    throw 'Python executable not found. Pass -PythonExe or create .venv before running this script.'
}

function Initialize-Workspace {
    foreach ($path in @($TempRoot, $ArtifactsDir, $FixturesDir, $HeadersDir, $ConfigsDir)) {
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
        }
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function New-Fixtures {
    param([string]$ResolvedPythonExe)

    $script = @'
from pathlib import Path
import sys

import numpy as np
from PIL import Image


root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)

size = 48
x = np.linspace(0, 255, size, dtype=np.uint8)
y = np.linspace(255, 0, size, dtype=np.uint8)
xx, yy = np.meshgrid(x, y)

sample = np.dstack([
    xx,
    yy,
    np.full_like(xx, 128),
])

fft_ref = np.dstack([
    np.roll(xx, 7, axis=1),
    np.full_like(xx, 196),
    np.roll(yy, 5, axis=0),
])

awb_ref = np.dstack([
    np.full_like(xx, 170),
    np.roll(xx, 3, axis=0),
    np.roll(yy, 9, axis=1),
])

Image.fromarray(sample).save(root / 'sample.png')
Image.fromarray(fft_ref).save(root / 'fft_ref.png')
Image.fromarray(awb_ref).save(root / 'awb_ref.png')

cube_lines = [
    'TITLE "api-test"',
    'LUT_3D_SIZE 2',
    'DOMAIN_MIN 0.0 0.0 0.0',
    'DOMAIN_MAX 1.0 1.0 1.0',
]

for r in (0.0, 1.0):
    for g in (0.0, 1.0):
        for b in (0.0, 1.0):
            cube_lines.append(f'{b:.6f} {g:.6f} {r:.6f}')

(root / 'identity.cube').write_text('\n'.join(cube_lines) + '\n', encoding='utf-8')
'@

    $script | & $ResolvedPythonExe - $FixturesDir
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to generate temporary API test fixtures.'
    }
}

function Get-HeaderValue {
    param(
        [string]$HeadersPath,
        [string]$HeaderName
    )

    if (-not (Test-Path $HeadersPath)) {
        return $null
    }

    $pattern = '^(?i)' + [regex]::Escape($HeaderName) + ':\s*(.+)$'
    $match = Get-Content $HeadersPath | Select-String -Pattern $pattern | Select-Object -Last 1
    if ($null -eq $match) {
        return $null
    }

    return $match.Matches[0].Groups[1].Value.Trim()
}

function Invoke-CurlRequest {
    param(
        [string]$Method,
        [string]$Url,
        [string]$BodyPath,
        [string]$HeadersPath,
        [string[]]$FormFields = @()
    )

    $args = @('-sS', '-X', $Method, '-D', $HeadersPath, '-o', $BodyPath, '-w', '%{http_code}')
    foreach ($field in $FormFields) {
        if ($field -match '^[^=]+=[@<]') {
            $args += @('-F', $field)
        }
        else {
            $args += @('--form-string', $field)
        }
    }
    $args += $Url

    $statusCode = & curl.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Method $Url"
    }

    return [int]$statusCode
}

function Test-HealthEndpoint {
    param([string]$TargetBaseUrl)

    $bodyPath = Join-Path $ArtifactsDir 'health.json'
    $headersPath = Join-Path $HeadersDir 'health.txt'
    try {
        $statusCode = Invoke-CurlRequest -Method 'GET' -Url "$TargetBaseUrl/health" -BodyPath $bodyPath -HeadersPath $headersPath
    }
    catch {
        return $false
    }

    if ($statusCode -ne 200) {
        return $false
    }

    $body = Get-Content -Raw $bodyPath | ConvertFrom-Json
    return $body.status -eq 'ok'
}

function Ensure-ServerAvailable {
    param([string]$ResolvedPythonExe)

    if (Test-HealthEndpoint -TargetBaseUrl $BaseUrl) {
        return
    }

    Write-Step 'FastAPI server not detected; starting run_api.py'

    $stdoutPath = Join-Path $TempRoot 'server.stdout.log'
    $stderrPath = Join-Path $TempRoot 'server.stderr.log'
    $serverProcess = Start-Process -FilePath $ResolvedPythonExe -ArgumentList 'run_api.py' -WorkingDirectory $WorkspaceRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    Set-Variable -Name serverProcess -Value $serverProcess -Scope Script

    foreach ($attempt in 1..30) {
        [System.Threading.Thread]::Sleep(1000)
        if ($serverProcess.HasExited) {
            $stderr = if (Test-Path $stderrPath) { Get-Content -Raw $stderrPath } else { '' }
            throw "run_api.py exited before becoming healthy. $stderr"
        }

        if (Test-HealthEndpoint -TargetBaseUrl $BaseUrl) {
            return
        }
    }

    throw 'Timed out waiting for FastAPI server to become healthy on /health.'
}

function ConvertTo-CompactJson {
    param([object]$Value)
    return ($Value | ConvertTo-Json -Depth 20 -Compress)
}

function Invoke-ImageTest {
    param(
        [string]$Name,
        [string]$Route,
        [string]$ExpectedStageOrder,
        [hashtable]$Config,
        [string[]]$ExtraFormFields = @()
    )

    $safeName = $Name.ToLowerInvariant().Replace(' ', '_')
    $bodyPath = Join-Path $ArtifactsDir "$safeName.png"
    $headersPath = Join-Path $HeadersDir "$safeName.txt"
    $configPath = Join-Path $ConfigsDir "$safeName.json"
    Set-Content -Path $configPath -Value (ConvertTo-CompactJson $Config) -Encoding UTF8 -NoNewline

    $formFields = @(
        "image=@$(Join-Path $FixturesDir 'sample.png')",
        "config=<$configPath",
        'output_format=png',
        'include_exif=true'
    ) + $ExtraFormFields

    $statusCode = Invoke-CurlRequest -Method 'POST' -Url "$BaseUrl$Route" -BodyPath $bodyPath -HeadersPath $headersPath -FormFields $formFields
    Assert-True ($statusCode -eq 200) "$Name failed with HTTP $statusCode"

    $contentType = Get-HeaderValue -HeadersPath $headersPath -HeaderName 'Content-Type'
    Assert-True ($null -ne $contentType -and $contentType.ToLowerInvariant().StartsWith('image/')) "$Name returned unexpected content type: $contentType"

    $stageCount = Get-HeaderValue -HeadersPath $headersPath -HeaderName 'X-Stage-Count'
    Assert-True ($stageCount -eq '1') "$Name returned unexpected X-Stage-Count: $stageCount"

    $stageOrder = Get-HeaderValue -HeadersPath $headersPath -HeaderName 'X-Stage-Order'
    Assert-True ($stageOrder -eq $ExpectedStageOrder) "$Name returned unexpected X-Stage-Order: $stageOrder"

    $outputSize = (Get-Item $bodyPath).Length
    Assert-True ($outputSize -gt 0) "$Name returned an empty body"
}

function Invoke-PipelineTest {
    param(
        [string]$Name,
        [hashtable]$Config,
        [string]$ExpectedStageOrder
    )

    $safeName = $Name.ToLowerInvariant().Replace(' ', '_')
    $bodyPath = Join-Path $ArtifactsDir "$safeName.png"
    $headersPath = Join-Path $HeadersDir "$safeName.txt"
    $configPath = Join-Path $ConfigsDir "$safeName.json"
    Set-Content -Path $configPath -Value (ConvertTo-CompactJson $Config) -Encoding UTF8 -NoNewline

    $statusCode = Invoke-CurlRequest -Method 'POST' -Url "$BaseUrl/api/v1/process/pipeline" -BodyPath $bodyPath -HeadersPath $headersPath -FormFields @(
        "image=@$(Join-Path $FixturesDir 'sample.png')",
        "fft_ref_image=@$(Join-Path $FixturesDir 'fft_ref.png')",
        "config=<$configPath",
        'output_format=png',
        'include_exif=true'
    )

    Assert-True ($statusCode -eq 200) "$Name failed with HTTP $statusCode"

    $stageCount = Get-HeaderValue -HeadersPath $headersPath -HeaderName 'X-Stage-Count'
    Assert-True ($stageCount -eq '3') "$Name returned unexpected X-Stage-Count: $stageCount"

    $stageOrder = Get-HeaderValue -HeadersPath $headersPath -HeaderName 'X-Stage-Order'
    Assert-True ($stageOrder -eq $ExpectedStageOrder) "$Name returned unexpected X-Stage-Order: $stageOrder"

    $outputSize = (Get-Item $bodyPath).Length
    Assert-True ($outputSize -gt 0) "$Name returned an empty body"
}

try {
    Write-Step 'Preparing curl API smoke test workspace'
    Initialize-Workspace

    Write-Step 'Resolving Python interpreter and generating fixtures'
    $resolvedPythonExe = Resolve-PythonExecutable
    New-Fixtures -ResolvedPythonExe $resolvedPythonExe

    Write-Step 'Ensuring FastAPI backend is available'
    Ensure-ServerAvailable -ResolvedPythonExe $resolvedPythonExe

    Write-Step 'Checking health endpoint'
    Assert-True (Test-HealthEndpoint -TargetBaseUrl $BaseUrl) 'Health endpoint did not return status=ok'

    $stageTests = @(
        [pscustomobject]@{ Name = 'Blend'; Route = '/api/v1/process/blend'; Stage = 'blend'; Config = @{}; Extra = @() },
        [pscustomobject]@{ Name = 'Non Semantic'; Route = '/api/v1/process/non-semantic'; Stage = 'non_semantic'; Config = @{ ns_iterations = 1; ns_learning_rate = 0.0003; ns_t_lpips = 0.04; ns_t_l2 = 0.00003; ns_c_lpips = 0.01; ns_c_l2 = 0.6; ns_grad_clip = 0.05 }; Extra = @() },
        [pscustomobject]@{ Name = 'CLAHE'; Route = '/api/v1/process/clahe'; Stage = 'clahe'; Config = @{}; Extra = @() },
        [pscustomobject]@{ Name = 'FFT'; Route = '/api/v1/process/fft'; Stage = 'fft'; Config = @{ fft_mode = 'ref'; fft_variant = 'v2'; seed = 7 }; Extra = @("fft_ref_image=@$(Join-Path $FixturesDir 'fft_ref.png')") },
        [pscustomobject]@{ Name = 'GLCM'; Route = '/api/v1/process/glcm'; Stage = 'glcm'; Config = @{ glcm_distances = @(1); glcm_angles = @(0.0, 0.7853981634); glcm_levels = 64; glcm_strength = 0.6; seed = 7 }; Extra = @("fft_ref_image=@$(Join-Path $FixturesDir 'fft_ref.png')") },
        [pscustomobject]@{ Name = 'LBP'; Route = '/api/v1/process/lbp'; Stage = 'lbp'; Config = @{ lbp_radius = 2; lbp_n_points = 16; lbp_method = 'uniform'; lbp_strength = 0.6; seed = 7 }; Extra = @("fft_ref_image=@$(Join-Path $FixturesDir 'fft_ref.png')") },
        [pscustomobject]@{ Name = 'Noise'; Route = '/api/v1/process/noise'; Stage = 'noise'; Config = @{ noise_std = 0.01; seed = 7 }; Extra = @() },
        [pscustomobject]@{ Name = 'Perturb'; Route = '/api/v1/process/perturb'; Stage = 'perturb'; Config = @{ perturb_magnitude = 0.003; seed = 7 }; Extra = @() },
        [pscustomobject]@{ Name = 'Sim Camera'; Route = '/api/v1/process/sim-camera'; Stage = 'sim_camera'; Config = @{ jpeg_cycles = 1; motion_blur_kernel = 1; seed = 7 }; Extra = @() },
        [pscustomobject]@{ Name = 'AWB'; Route = '/api/v1/process/awb'; Stage = 'awb'; Config = @{ seed = 7 }; Extra = @("ref_image=@$(Join-Path $FixturesDir 'awb_ref.png')") },
        [pscustomobject]@{ Name = 'LUT'; Route = '/api/v1/process/lut'; Stage = 'lut'; Config = @{ lut_strength = 0.2 }; Extra = @("lut_file=@$(Join-Path $FixturesDir 'identity.cube')") }
    )

    foreach ($test in $stageTests) {
        if ($SkipStages -contains $test.Stage) {
            Write-Step "Skipping $($test.Stage)"
            continue
        }

        Write-Step "Running $($test.Name)"
        Invoke-ImageTest -Name $test.Name -Route $test.Route -ExpectedStageOrder $test.Stage -Config $test.Config -ExtraFormFields $test.Extra
    }

    Write-Step 'Running pipeline default-order coverage'
    Invoke-PipelineTest -Name 'Pipeline Default Order' -ExpectedStageOrder 'clahe,fft,noise' -Config @{
        execution_order = $false
        stages = @{
            noise = @{ noise_std = 0.01; seed = 11 }
            clahe = @{ clahe_clip = 2.0; tile = 8 }
            fft = @{ fft_mode = 'ref'; fft_variant = 'v2'; seed = 11 }
        }
    }

    Write-Step 'Running pipeline explicit-order coverage'
    Invoke-PipelineTest -Name 'Pipeline Explicit Order' -ExpectedStageOrder 'noise,fft,clahe' -Config @{
        execution_order = $true
        stage_order = @('noise', 'fft', 'clahe')
        stages = @{
            noise = @{ noise_std = 0.01; seed = 13 }
            clahe = @{ clahe_clip = 2.0; tile = 8 }
            fft = @{ fft_mode = 'ref'; fft_variant = 'v2'; seed = 13 }
        }
    }

    Write-Host ''
    Write-Host 'All curl API smoke tests passed.' -ForegroundColor Green
    Write-Host "Artifacts: $ArtifactsDir"
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
    }

    if (-not $KeepArtifacts -and (Test-Path $TempRoot)) {
        Remove-Item -Recurse -Force $TempRoot
    }
}