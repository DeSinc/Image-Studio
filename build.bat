@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Building release binary...
cargo build --release
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed!
    timeout /t 3 2>nul
    pause
    exit /b %ERRORLEVEL%
)

echo [2/2] Updating ImageStudio.exe in workspace root...
copy /y "target\release\ImageStudio.exe" "ImageStudio.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Could not overwrite root ImageStudio.exe - it might be currently running.
    echo You can still find the built executable at: target\release\ImageStudio.exe
) else (
    echo [SUCCESS] Build complete! Executable updated at ImageStudio.exe
)

echo.
timeout /t 3 2>nul

