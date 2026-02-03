@echo off
setlocal

set REQ_FILE=%~dp0requirements\requirements.txt

echo Exporting installed packages to %REQ_FILE%
pip freeze > "%REQ_FILE%"

echo Done!
endlocal
