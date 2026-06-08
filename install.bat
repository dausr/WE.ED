@echo off
title WE.ED.CUT.IT.CLAW Installer
echo ==================================================
echo         WE.ED.CUT.IT.CLAW Setup
echo ==================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
echo ==================================================
echo Installation finished. Press any key to exit.
echo ==================================================
if /i not "%WEEDIT_NO_PAUSE%"=="1" pause > nul
