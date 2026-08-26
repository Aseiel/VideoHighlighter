@echo off
REM Isolated bootstrap entry point — does not change the app or CI.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-VideoHighlighter.ps1" %*
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo.
  echo Install failed with exit code %ERR%.
  pause
  exit /b %ERR%
)
echo.
pause
