@echo off
rem ============================================================
rem  ComfyUI model setup — just double-click this file.
rem  It quietly fetches its own tools; nothing to install first.
rem ============================================================
setlocal
cd /d "%~dp0"
title ComfyUI Setup
echo.
echo  Starting the ComfyUI setup wizard...
echo  (a browser window will open in a moment — leave THIS window open)
echo.

rem --- make sure uv (a tiny portable Python manager) is available ---
set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo  First run: fetching a small helper tool ^(about 20 MB^)...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
          "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>nul
        set "UV=%USERPROFILE%\.local\bin\uv.exe"
    )
)
if not exist "%UV%" if "%UV%" neq "uv" (
    echo  Could not fetch the helper tool. Are you connected to the internet?
    echo  If this keeps happening, see TROUBLESHOOTING.md.
    pause
    exit /b 1
)

"%UV%" run --python 3.12 provision.py wizard
if errorlevel 1 (
    echo.
    echo  Setup closed with a problem. Check the logs folder for details,
    echo  or just run Setup again — downloads continue where they left off.
    pause
)
endlocal
