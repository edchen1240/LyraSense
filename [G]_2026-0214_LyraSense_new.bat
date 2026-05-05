@echo off
setlocal EnableDelayedExpansion

REM ==================================================
REM SETTINGS
REM ==================================================
set "WORKDIR=D:\01_Floor\a_Ed\09_EECS\10_Python\03_Developing\2026-0214_LyraSense"
set "TARGET_DIR=."
set "REPO_URL=git@github.com:edchen1240/LyraSense.git"
set "BRANCH=main"
set "COMMIT_MSG=Auto commit from batch script"

REM ==================================================
REM PULL STRATEGY - Choose one:
REM ==================================================
REM (default)   : git pull with automatic merge if needed
REM              Use for: Personal projects, auto-sync workflows
REM              Creates merge commits, handles conflicts gracefully
REM
REM --rebase    : Linear history, replay local commits on top of remote
REM              Use for: Clean linear history, squashing commits
REM              May fail if branches diverged (requires manual conflict resolution)
REM
REM --ff-only   : Only pulls if it's a fast-forward (no conflicts)
REM              Use for: Strict linear repos, CI/CD pipelines
REM              Fails if branches diverged (safer for automation)
REM ==================================================
set "PULL_STRATEGY="
REM ==================================================

:MENU
cls
echo ============================================
echo GitHub Sync Script
echo ============================================
echo Working directory : %WORKDIR%
echo Repo folder       : %TARGET_DIR%
echo Remote repo       : %REPO_URL%
echo Branch            : %BRANCH%
echo.
echo Choose operation:
echo   1 - Pull
echo   2 - Push
echo   3 - Both (Pull then Push)
echo   Q - Quit
echo.

set /p CHOICE=Enter choice: 

if /i "%CHOICE%"=="1" (
    call :DoPull
    goto :END
)

if /i "%CHOICE%"=="2" (
    call :DoPush
    goto :END
)

if /i "%CHOICE%"=="3" (
    call :DoPull
    if errorlevel 1 goto :FAIL
    call :DoPush
    goto :END
)

if /i "%CHOICE%"=="Q" exit /b

echo Invalid choice.
pause
goto MENU


REM ==================================================
REM Pull operation
REM ==================================================
:DoPull
echo.
echo [PULL] Starting pull operation...

if not exist "%WORKDIR%" (
    echo Creating working directory...
    mkdir "%WORKDIR%"
)

cd /d "%WORKDIR%"

if not exist "%TARGET_DIR%" (
    echo Local repo not found. Cloning from GitHub...
    git clone -b %BRANCH% "%REPO_URL%" "%TARGET_DIR%" || goto :FAIL
    exit /b 0
)

cd /d "%WORKDIR%\%TARGET_DIR%"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo ERROR: Directory exists but is not a Git repository.
    goto :FAIL
)

echo Fetching latest changes...
git fetch origin || goto :FAIL

echo Pulling updates...
git pull %PULL_STRATEGY% --allow-unrelated-histories origin %BRANCH% || goto :FAIL

exit /b 0


REM ==================================================
REM Push operation
REM ==================================================
:DoPush
echo.
echo [PUSH] Starting push operation...

set "REPO_PATH=%WORKDIR%\%TARGET_DIR%"

if not exist "%REPO_PATH%" (
    echo ERROR: Repo directory not found.
    goto :FAIL
)

cd /d "%REPO_PATH%"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo ERROR: Not a Git repository.
    goto :FAIL
)

echo Current status:
git status

echo.
echo Staging files...
git add -A

echo.
echo Committing...
git commit -m "%COMMIT_MSG%" >nul 2>&1

if errorlevel 1 (
    echo Nothing to commit.
)

echo.
echo Pushing to GitHub...
git push origin %BRANCH% || goto :FAIL

exit /b 0


REM ==================================================
REM Failure handler
REM ==================================================
:FAIL
echo.
echo ERROR occurred.
pause
exit /b 1

:END
echo.
echo Done.          (%TARGET_DIR%)
pause
goto MENU