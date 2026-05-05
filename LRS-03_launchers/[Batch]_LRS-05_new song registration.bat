<<<<<<< HEAD
@echo off
set "dir_current=%~dp0"

:loop
cd /d "D:\01_Floor\a_Ed\09_EECS\10_Python\03_Developing\2026-0214_LyraSense"
python "LRS-05_new song registration.py" "%dir_current%"

echo.
echo Press ENTER to run again, or Ctrl+C to exit.
pause >nul
goto loop

=======
@echo off
set "dir_current=%~dp0"

:loop
cd /d "D:\01_Floor\a_Ed\09_EECS\10_Python\03_Developing\2026-0214_LyraSense"
python "LRS-05_new song registration.py" "%dir_current%"

echo.
echo Press ENTER to run again, or Ctrl+C to exit.
pause >nul
goto loop

>>>>>>> 61910d571079a324c71ca9f2f536cc37805e49f4
