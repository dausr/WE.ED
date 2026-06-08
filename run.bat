@echo off
title WE.ED.CUT.IT.CLAW Runner
echo ==================================================
echo         WE.ED.CUT.IT.CLAW Runner
echo ==================================================
echo.
python "%~dp0beat_sync.py" %*
echo.
echo ==================================================
echo Execution completed. Press any key to exit.
echo ==================================================
if /i not "%WEEDIT_NO_PAUSE%"=="1" pause > nul
