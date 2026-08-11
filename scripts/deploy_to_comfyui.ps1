# Deploy validated GITS source into a ComfyUI custom_nodes folder.
#
# Usage (from package root):
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_to_comfyui.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_to_comfyui.ps1 -Target "D:\ComfyUI\custom_nodes\ComfyUI-GITS-Tracked-Identity"
#
# Resolution order for -Target when omitted:
#   1. Environment variable GITS_COMFYUI_CUSTOM_NODES
#   2. Sibling path ..\..\custom_nodes\ComfyUI-GITS-Tracked-Identity (if this repo lives under custom_nodes)
#   3. Explicit -Target is required otherwise

param(
    [string]$Target = ""
)

$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $Source)) {
    throw "Source package not found: $Source"
}

if (-not $Target) {
    if ($env:GITS_COMFYUI_CUSTOM_NODES) {
        $Target = $env:GITS_COMFYUI_CUSTOM_NODES
    }
    elseif ((Split-Path -Leaf (Split-Path -Parent $Source)) -ieq "custom_nodes") {
        $Target = $Source
    }
    else {
        throw "Pass -Target `"C:\path\to\ComfyUI\custom_nodes\ComfyUI-GITS-Tracked-Identity`" or set GITS_COMFYUI_CUSTOM_NODES."
    }
}

Write-Host "Source: $Source"
Write-Host "Target: $Target"

New-Item -ItemType Directory -Force -Path $Target | Out-Null

$include = @(
    "__init__.py",
    "pyproject.toml",
    "requirements.txt",
    "LICENSE",
    "README.md",
    "CHANGELOG.md"
)

foreach ($name in $include) {
    $src = Join-Path $Source $name
    if (Test-Path $src) {
        Copy-Item -Force $src (Join-Path $Target $name)
        Write-Host "  copied $name"
    }
}

foreach ($dir in @("nodes", "assets", "workflows", "scripts", "models")) {
    $srcDir = Join-Path $Source $dir
    if (-not (Test-Path $srcDir)) { continue }
    $dstDir = Join-Path $Target $dir
    New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    Get-ChildItem -Path $srcDir -Recurse -File | Where-Object {
        $_.FullName -notmatch '\\__pycache__\\' -and
        $_.Extension -ne '.pyc' -and
        $_.Name -ne '.gitkeep'
    } | ForEach-Object {
        $rel = $_.FullName.Substring($srcDir.Length).TrimStart('\', '/')
        $dest = Join-Path $dstDir $rel
        $destParent = Split-Path -Parent $dest
        if (-not (Test-Path $destParent)) {
            New-Item -ItemType Directory -Force -Path $destParent | Out-Null
        }
        Copy-Item -Force $_.FullName $dest
    }
    Write-Host "  synced $dir/"
}

Get-ChildItem -Path $Target -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Deploy complete. Fully restart ComfyUI to load Python node changes."
Write-Host "Models live under ComfyUI\models\gits_tracking (not overwritten by this script)."
