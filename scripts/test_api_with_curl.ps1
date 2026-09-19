#Requires -Version 7
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ImagePath,

    [string]$BaseUrl = 'http://127.0.0.1:8000',

    [ValidateSet('A', 'B')]
    [string]$Pipeline = 'A',

    [switch]$KeepResponses
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Invoke-ComposeRequest {
    param(
        [string]$Url,
        [string[]]$FormFields,
        [string]$OutPath,
        [string]$HeadersPath,
        [string]$ExpectContentTypePrefix,
        [string]$StepName
    )

    $curlArgs = @('-sS', '--fail-with-body', '-D', $HeadersPath, '-o', $OutPath)
    foreach ($field in $FormFields) {
        if ($field -match '^image=@|^reference_image=@|^lut_file=@') {
            $curlArgs += @('-F', $field)
        }
        else {
            # JSON config values go as literal strings so quoting stays safe on Windows.
            $curlArgs += @('--form-string', $field)
        }
    }
    $curlArgs += $Url

    Write-Step $StepName
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Url (exit code $LASTEXITCODE)"
    }

    if (-not (Test-Path -LiteralPath $OutPath -PathType Leaf)) {
        throw "No response body written for $Url"
    }
    if ((Get-Item -LiteralPath $OutPath).Length -le 0) {
        throw "Empty response body from $Url"
    }

    $headerMatch = Get-Content -LiteralPath $HeadersPath |
        Select-String -Pattern '^(?i)Content-Type:\s*(.+)$' |
        Select-Object -Last 1
    $contentType = if ($null -ne $headerMatch) { $headerMatch.Matches[0].Groups[1].Value.Trim() } else { $null }
    if ($null -eq $contentType -or -not $contentType.ToLowerInvariant().StartsWith($ExpectContentTypePrefix)) {
        throw "Unexpected Content-Type from ${Url}: $contentType"
    }

    Write-Host "    Content-Type: $contentType"
    Write-Host "    Response file: $OutPath"
}

function New-SampleLut {
    param([string]$Path)

    $lines = @(
        'TITLE "api-test-identity"',
        'LUT_3D_SIZE 2',
        'DOMAIN_MIN 0.0 0.0 0.0',
        'DOMAIN_MAX 1.0 1.0 1.0'
    )
    foreach ($b in 0.0, 1.0) {
        foreach ($g in 0.0, 1.0) {
            foreach ($r in 0.0, 1.0) {
                $lines += ('{0:F6} {1:F6} {2:F6}' -f $b, $g, $r)
            }
        }
    }
    [System.IO.File]::WriteAllText($Path, ($lines -join "`n") + "`n")
}

# Client-composition smoke test: each endpoint receives the raw binary
# response of the previous endpoint as its "image" upload, so the ordering is
# owned entirely by this client script. The forensic-camera JPEG finalizer
# is invoked last in both pipelines.

$WorkspaceDir = Join-Path ([System.IO.Path]::GetTempPath()) ('curl-compose-' + [guid]::NewGuid().ToString('N').Substring(0, 8))

try {
    New-Item -ItemType Directory -Path $WorkspaceDir | Out-Null

    $resolvedInput = (Resolve-Path -LiteralPath $ImagePath).Path
    Write-Step "Validated input image: $resolvedInput"
    Write-Step "Base URL: $BaseUrl"
    Write-Step "Pipeline: $Pipeline"

    Write-Step 'Health check'
    & curl.exe -sS --fail-with-body -o (Join-Path $WorkspaceDir 'health.json') "$BaseUrl/health"
    if ($LASTEXITCODE -ne 0) {
        throw "Health check failed against $BaseUrl/health (is the server running?)"
    }
    $health = Get-Content -Raw (Join-Path $WorkspaceDir 'health.json') | ConvertFrom-Json
    if ($health.status -ne 'ok') {
        throw 'Health endpoint did not report status=ok'
    }

    $lutPath = $null
    $pipelineSteps = switch ($Pipeline) {
        'A' {
            # Order: clahe -> noise -> color-blend -> forensic-camera
            @(
                [pscustomobject]@{
                    Name = 'CLAHE normalization'
                    Url = "$BaseUrl/api/v1/clahe"
                    Form = @('config={"clahe_clip":2.0,"tile":8}', 'output_format=png')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'Gaussian noise'
                    Url = "$BaseUrl/api/v1/noise"
                    Form = @('config={"noise_std":0.01,"seed":123}', 'output_format=png')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'Color-region blending'
                    Url = "$BaseUrl/api/v1/color-blend"
                    Form = @('output_format=png', 'include_exif=true')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'Forensic camera finalizer'
                    Url = "$BaseUrl/api/v1/forensic-camera"
                    Form = @('config={"profile":"iphone_16_pro","software":"17.1.2","seed":7}')
                    ExpectPrefix = 'image/jpeg'
                    Ext = 'jpg'
                }
            )
        }
        'B' {
            $lutPath = Join-Path $WorkspaceDir 'identity.cube'
            New-SampleLut -Path $lutPath

            # Order: perturb -> lut -> awb -> glcm -> forensic-camera
            @(
                [pscustomobject]@{
                    Name = 'Randomized perturbation'
                    Url = "$BaseUrl/api/v1/perturb"
                    Form = @('config={"perturb_magnitude":0.003,"seed":11}', 'output_format=png')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'Apply identity LUT'
                    Url = "$BaseUrl/api/v1/lut"
                    Form = @("lut_file=@$lutPath", 'config={"lut_strength":0.2}', 'output_format=png')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'Auto white balance'
                    Url = "$BaseUrl/api/v1/awb"
                    Form = @('output_format=png')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'GLCM normalization'
                    Url = "$BaseUrl/api/v1/glcm"
                    Form = @('config={"glcm_distances":[1],"glcm_angles":[0.0],"glcm_levels":64,"glcm_strength":0.6,"seed":13}', 'output_format=png')
                    ExpectPrefix = 'image/'
                    Ext = 'png'
                },
                [pscustomobject]@{
                    Name = 'Forensic camera finalizer'
                    Url = "$BaseUrl/api/v1/forensic-camera"
                    Form = @('config={"profile":"iphone_16_pro","software":"14.0","seed":9}')
                    ExpectPrefix = 'image/jpeg'
                    Ext = 'jpg'
                }
            )
        }
    }

    $currentImage = $resolvedInput
    $index = 0

    foreach ($step in $pipelineSteps) {
        $safeName = ($step.Name.ToLowerInvariant() -replace '[^a-z0-9]+', '_').Trim('_')
        $outPath = Join-Path $WorkspaceDir ("{0:D2}_{1}.{2}" -f $index, $safeName, $step.Ext)
        $headersPath = Join-Path $WorkspaceDir ("{0:D2}_{1}.headers.txt" -f $index, $safeName)

        # Every request gets the previous response as its "image" upload.
        $formFields = @("image=@$currentImage") + @($step.Form)

        Invoke-ComposeRequest -Url $step.Url -FormFields $formFields -OutPath $outPath `
            -HeadersPath $headersPath -ExpectContentTypePrefix $step.ExpectPrefix `
            -StepName ("{0} -> {1}" -f $step.Name, ($step.Url -replace [regex]::Escape($BaseUrl), ''))

        $currentImage = $outPath
        $index++
    }

    Write-Step 'Composition finished; response files preserved:'
    Write-Host "Final output: $currentImage"
    Write-Host "Workspace:    $WorkspaceDir"
    if (-not $KeepResponses) {
        Write-Host 'Pass -KeepResponses to preserve the workspace; otherwise it is removed on exit.'
    }
}
finally {
    if (-not $KeepResponses -and (Test-Path -LiteralPath $WorkspaceDir)) {
        Remove-Item -LiteralPath $WorkspaceDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
