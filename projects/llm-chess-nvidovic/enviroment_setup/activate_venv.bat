@echo off
set VENV_DIR=%~dp0..\venv

if not exist "%VENV_DIR%" (
    echo No venv found. Run setup_venv.bat first.
    exit /b 1
)

echo Activating environment...
call "%VENV_DIR%\Scripts\activate"

cmd /k
