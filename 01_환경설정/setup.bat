@echo off
setlocal
echo ==========================================
echo  BigData Analyst (Practical) - Env Setup
echo ==========================================
echo.

set "HERE=%~dp0"
cd /d "%HERE%.."
echo Project folder: %CD%
echo.

rem "where python" also matches the Microsoft Store stub, so actually run it
set "PY=python"
python -c "import sys" >nul 2>nul
if errorlevel 1 (
    set "PY=py -3"
    py -3 -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] python not found.
        echo         Reinstall from python.org and CHECK "Add python.exe to PATH".
        pause
        exit /b 1
    )
)
%PY% --version

if not exist ".venv" (
    echo.
    echo [1/3] Creating virtual env ^(.venv^) ...
    %PY% -m venv .venv
) else (
    echo.
    echo [1/3] .venv already exists - skip
)

echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q

echo [3/3] Installing packages (1-3 min) ...
".venv\Scripts\python.exe" -m pip install -r "%HERE%requirements.txt"

echo.
echo ==========================================
echo  Check
echo ==========================================
for %%f in ("%HERE%*.py") do ".venv\Scripts\python.exe" "%%~f"

echo.
echo DONE. Open this folder in VS Code and run
echo   Ctrl+Shift+P  ^>  "Python: Select Interpreter"  ^>  choose .venv
echo.
pause
