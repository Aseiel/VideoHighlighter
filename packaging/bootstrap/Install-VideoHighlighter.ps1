#Requires -Version 5.1
<#
.SYNOPSIS
  Download both Windows 7z volumes for VideoHighlighter and extract them.

.DESCRIPTION
  Isolated bootstrap - does not modify the app or CI.
  Reads config.json next to this script. Downloads into out\download\,
  extracts into out\VideoHighlighter\.

.PARAMETER ConfigPath
  Optional path to config.json (default: next to this script).

.PARAMETER SkipExtract
  Download only; do not run 7z.
#>
[CmdletBinding()]
param(
    [string]$ConfigPath = "",
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $Root "config.json"
}

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-SevenZip {
    $cmd = Get-Command 7z -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $tools = Join-Path $Root "out\tools"
    $local = Join-Path $tools "7zr.exe"
    if (Test-Path $local) { return $local }

    Write-Step "7-Zip not on PATH - downloading 7zr.exe (official LZMA SDK stub)"
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    # Official 7-Zip extra: standalone 7zr (https://www.7-zip.org/download.html)
    $url = "https://www.7-zip.org/a/7zr.exe"
    Invoke-WebRequest -Uri $url -OutFile $local -UseBasicParsing
    if (-not (Test-Path $local)) {
        throw "Could not download 7zr.exe. Install 7-Zip and ensure 7z is on PATH."
    }
    return $local
}

function Get-FileWithProgress {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutFile
    )
    $dir = Split-Path -Parent $OutFile
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    if (Test-Path $OutFile) {
        $existing = (Get-Item $OutFile).Length
        Write-Host ("  already present: {0} ({1} bytes) - skipping download" -f $OutFile, $existing)
        return
    }

    Write-Host "  GET $Uri"
    # BitsTransfer is nicer for multi-GB; fall back to Invoke-WebRequest.
    try {
        Import-Module BitsTransfer -ErrorAction Stop
        Start-BitsTransfer -Source $Uri -Destination $OutFile -DisplayName "VideoHighlighter download"
    }
    catch {
        Write-Host "  BITS unavailable, using Invoke-WebRequest..."
        Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
    }

    if (-not (Test-Path $OutFile) -or (Get-Item $OutFile).Length -lt 1MB) {
        throw "Download failed or file too small: $OutFile"
    }
}

# The same file from each of $Uris in turn (the mirror, then GitHub), until one
# works. A failed attempt's partial file is removed, or the next run would take
# it for a finished download.
function Get-FileFromAny {
    param(
        [Parameter(Mandatory = $true)][string[]]$Uris,
        [Parameter(Mandatory = $true)][string]$OutFile
    )
    $last = $null
    foreach ($uri in $Uris) {
        try {
            Get-FileWithProgress -Uri $uri -OutFile $OutFile
            return
        }
        catch {
            $last = $_
            Write-Warning "  failed: $uri ($($_.Exception.Message))"
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
        }
    }
    throw $last
}

# --- main ---
Write-Host "VideoHighlighter bootstrap installer (isolated)"
Write-Host "Config: $ConfigPath"

if (-not (Test-Path $ConfigPath)) {
    throw "config.json not found: $ConfigPath"
}

$config = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
$downloadDir = Join-Path $Root "out\download"
$extractDir = Join-Path $Root "out\$($config.product_name)"

$tag = $config.tag
$base = $config.base_url.TrimEnd("/")
$assets = @($config.assets)
$pattern = $config.asset_pattern
# The download mirror (tools/mirror_release.py): <mirror>/<tag>/<asset>, and
# <mirror>/latest/<edition>.json naming the newest release. Empty when the
# build had none; older configs lack the field.
$mirror = ""
if ($config.PSObject.Properties["mirror"] -and $config.mirror) {
    $mirror = ([string]$config.mirror).TrimEnd("/")
}

# A pinned tag goes stale the moment a release ships, and the installer then
# hands out an old build for as long as nobody edits this file. Ask the API
# which release is current and take the volumes from it, or the mirror when
# the API cannot be reached; the pinned values in config.json stay as the
# offline fallback.
if ($config.use_latest) {
    $resolved = $false
    try {
        Write-Step "Resolving latest release of $($config.repo)"
        $api = "https://api.github.com/repos/$($config.repo)/releases/latest"
        $rel = Invoke-RestMethod -Uri $api -UseBasicParsing -Headers @{
            "User-Agent" = "VideoHighlighter-bootstrap"
            "Accept"     = "application/vnd.github+json"
        }
        $found = @($rel.assets | Where-Object { $_.name -match $pattern } |
            Sort-Object name)
        if ($found.Count -gt 0) {
            $tag = $rel.tag_name
            $assets = $found | ForEach-Object { $_.name }
            $base = ($found[0].browser_download_url -replace '/[^/]+$', '')
            $resolved = $true
            Write-Host "  latest is $tag ($($assets.Count) matching parts)"
        }
        else {
            Write-Warning "No asset matched '$pattern' in $($rel.tag_name)"
        }
    }
    catch {
        Write-Warning "Could not reach the GitHub API ($($_.Exception.Message))"
    }
    if (-not $resolved -and $mirror) {
        try {
            Write-Step "Asking the download mirror for the latest release"
            $index = Invoke-RestMethod -UseBasicParsing `
                -Uri "$mirror/latest/$(([string]$config.edition).ToLower()).json"
            $found = @($index.assets | Where-Object { $_.name -match $pattern } |
                Sort-Object name)
            if ($found.Count -gt 0) {
                $tag = $index.tag
                $assets = $found | ForEach-Object { $_.name }
                $base = "https://github.com/$($config.repo)/releases/download/$tag"
                $resolved = $true
                Write-Host "  latest is $tag ($($assets.Count) matching parts)"
            }
        }
        catch {
            Write-Warning "Could not reach the download mirror either ($($_.Exception.Message))"
        }
    }
    if (-not $resolved) {
        Write-Warning "Using the pinned $tag"
    }
}

Write-Step "Downloading $($config.edition) $tag ($($assets.Count) parts)"
$localParts = @()
foreach ($name in $assets) {
    $sources = @()
    if ($mirror) { $sources += "$mirror/$tag/$name" }
    $sources += "$base/$name"
    $dest = Join-Path $downloadDir $name
    Get-FileFromAny -Uris $sources -OutFile $dest
    $localParts += $dest
}

if ($SkipExtract) {
    Write-Step "SkipExtract set - done. Files in $downloadDir"
    exit 0
}

$sevenZip = Get-SevenZip
Write-Step "Extracting with $sevenZip"

# Split archives: pass the .001 volume; 7z finds the rest in the same folder.
$first = $localParts | Where-Object { $_ -match '\.001$' } | Select-Object -First 1
if (-not $first) {
    throw "No .001 volume among the downloaded files - nothing to extract."
}

# 7z reads the later volumes off disk by name, so a gap surfaces as a confusing
# archive error rather than a missing download. Say which one is missing.
for ($i = 1; $i -le $localParts.Count; $i++) {
    $volume = ($first -replace '\.\d{3}$', ('.{0:D3}' -f $i))
    if (-not (Test-Path $volume)) {
        throw "Missing archive volume: $volume"
    }
}

New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
& $sevenZip x -y "-o$extractDir" $first
if ($LASTEXITCODE -ne 0) {
    throw "7z extract failed with exit code $LASTEXITCODE"
}

$exe = Get-ChildItem -Path $extractDir -Recurse -Filter "$($config.product_name).exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1

Write-Step "Done"
if ($exe) {
    Write-Host "Installed:"
    Write-Host "  $($exe.FullName)"
    $revealPath = $exe.DirectoryName
}
else {
    Write-Warning "Extracted, but $($config.product_name).exe was not found under $extractDir"
    $revealPath = $extractDir
}
Write-Host ""
Write-Host "You can delete out\download\ later to free disk space."

try {
    Start-Process explorer.exe $revealPath
}
catch {
    # non-fatal
}
