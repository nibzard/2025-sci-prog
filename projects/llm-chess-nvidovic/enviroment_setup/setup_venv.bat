@echo off
setlocal

REM ---- PATH SETTINGS ----
set ROOT_DIR=%~dp0..
set VENV_DIR=%ROOT_DIR%\venv
set REQ_DIR=%~dp0requirements
set REQ_FILE=%REQ_DIR%\requirements.txt

echo Project root: %ROOT_DIR%
echo Venv path: %VENV_DIR%
echo Requirements: %REQ_FILE%
echo.

REM ---- CREATE VENV ----
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM ---- UPDATE PIP ----
echo Upgrading pip...
call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip

REM ---- INSTALL REQUIREMENTS ----
if exist "%REQ_FILE%" (
    echo Installing requirements...
    call "%VENV_DIR%\Scripts\pip.exe" install -r "%REQ_FILE%"
) else (
    echo No requirements.txt found.
)

REM ---- AUTO FREEZE BACK ----
echo Saving installed packages back to requirements.txt...
call "%VENV_DIR%\Scripts\pip.exe" freeze > "%REQ_FILE%"

echo.
echo Setup complete!
echo Activate with:
echo     enviroment_setup\activate_venv.bat

endlocal
exit /b 0
