# Launches the Tauri dev app with the MSVC build environment loaded.
# Rust on Windows needs the VS Build Tools CRT/linker on PATH/LIB; this repo's
# default shell doesn't have it, so we source vcvars64.bat first.
$ErrorActionPreference = "Stop"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
  # Fall back to vswhere discovery if the path changes.
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  $vcvars = Join-Path $inst "VC\Auxiliary\Build\vcvars64.bat"
}
Push-Location $PSScriptRoot
cmd /c "`"$vcvars`" >nul 2>&1 && pnpm tauri dev"
Pop-Location
