@echo off
set "dir_current=%~dp0"

:loop
cd /d "D:\01_Floor\a_Ed\09_EECS\10_Python\03_Developing\2026-0214_LyraSense"
python "LRS-12_DTW matching.py" "%dir_current%"

echo.
echo Press ENTER to run again, or Ctrl+C to exit.
pause >nul
goto loop

