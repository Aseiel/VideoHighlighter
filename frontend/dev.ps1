# Launches the Tauri dev app.
#
# Two things this handles that a bare `pnpm tauri dev` does not:
#
# 1. MSVC environment. Rust on Windows needs the VS Build Tools CRT/linker on
#    PATH/LIB; a normal shell doesn't have them, and without this you get
#    "LNK1104: cannot open file 'msvcrt.lib'" -- which looks like a Rust problem
#    but isn't.
# 2. Stale processes. Vite's port is strict (5173) and the sidecar's is fixed
#    (8756), so anything left over from a crashed or backgrounded run makes the
#    launch die with "Port 5173 is already in use". Clear them first.
#
# Usage:  .\dev.ps1           launch
#         .\dev.ps1 -Stop     just clean up, don't launch

param([switch]$Stop)

$ErrorActionPreference = "Stop"

function Stop-StaleProcesses {
    # The Tauri app window itself.
    Get-Process -Name "video-highlighter" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host "  stopping app (pid $($_.Id))"
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }

    # Vite / the tauri CLI.
    Get-CimInstance Win32_Process -Filter "Name='node.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*vite*' -or $_.CommandLine -like '*tauri*' } |
        ForEach-Object {
            Write-Host "  stopping node (pid $($_.ProcessId))"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }

    # The Python sidecar, run from source (dev) or as the packaged exe.
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*sidecar.server*' } |
        ForEach-Object {
            Write-Host "  stopping sidecar (pid $($_.ProcessId))"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Get-Process -Name "vh-sidecar" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host "  stopping vh-sidecar (pid $($_.Id))"
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }

    Start-Sleep -Seconds 1

    # Anything else holding our ports isn't ours to kill -- say so rather than
    # letting the user hit a confusing failure a minute into a Rust compile.
    foreach ($port in 5173, 8756) {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            $owner = (Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue).ProcessName
            Write-Warning "Port $port is still in use by '$owner' (pid $($conn.OwningProcess)) -- not mine to stop. Close it, then re-run."
            exit 1
        }
    }
}

Write-Host "Cleaning up stale dev processes..."
Stop-StaleProcesses

if ($Stop) {
    Write-Host "Done. Nothing running."
    exit 0
}

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    # Fall back to vswhere discovery if the install path differs.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error "Visual Studio Build Tools not found. Rust needs the MSVC toolchain: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        exit 1
    }
    $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $inst) {
        Write-Error "vswhere found no install with the C++ build tools. Install the 'Desktop development with C++' workload."
        exit 1
    }
    $vcvars = Join-Path $inst "VC\Auxiliary\Build\vcvars64.bat"
}

Write-Host "Starting (first run compiles Rust, ~90s)..."
Push-Location $PSScriptRoot
try {
    # Build the cmd line in a variable rather than inline: Windows PowerShell
    # 5.1 parses a literal '&&' inside a double-quoted argument as a statement
    # separator and fails at *parse* time, so the whole script won't even load.
    # (PowerShell 7 accepts it, which makes this easy to miss.) cmd.exe still
    # gets the '&&' it needs -- vcvars must set the env in the same shell that
    # then runs pnpm.
    $cmdLine = '"' + $vcvars + '" >nul 2>&1 && pnpm tauri dev'
    & cmd.exe /c $cmdLine
} finally {
    Pop-Location
}
