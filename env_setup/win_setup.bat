@echo off
setlocal EnableDelayedExpansion
set LOGFILE=setup_log.txt
echo Setting up environment for Windows... > %LOGFILE%

:: Step 1: Check if Ruby is installed
echo Checking if Ruby is installed...
echo Checking if Ruby is installed... >> %LOGFILE%
ruby -v > nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo Ruby is already installed.
    echo Ruby is already installed. >> %LOGFILE%
    ruby -v >> %LOGFILE%
) else (
    echo Ruby is not installed. Installing...
    echo Ruby is not installed. Installing... >> %LOGFILE%
    powershell -Command "Invoke-WebRequest -Uri https://github.com/oneclick/rubyinstaller2/releases/download/RubyInstaller-3.1.3-1/rubyinstaller-devkit-3.1.3-1-x64.exe -OutFile rubyinstaller-devkit.exe"
    start /wait rubyinstaller-devkit.exe
    pause
)

:: Step 2: Check if Jekyll is installed
echo Checking if Jekyll is installed...
echo Checking if Jekyll is installed... >> %LOGFILE%
call jekyll -v >nul 2>&1
set JEKYLL_CHECK=!ERRORLEVEL!
if !JEKYLL_CHECK! EQU 0 (
    echo Jekyll is already installed.
    echo Jekyll is already installed. >> %LOGFILE%
    call jekyll -v >> %LOGFILE%
) else (
    echo Jekyll is not installed or not found. Installing...
    echo Jekyll is not installed or not found. Installing... >> %LOGFILE%
    call gem install jekyll >> %LOGFILE% 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo Failed to install Jekyll.
        echo Failed to install Jekyll. >> %LOGFILE%
        pause
        exit /b 1
    )
)

:: Step 3: Check if Bundler is installed
echo Checking if Bundler is installed...
echo Checking if Bundler is installed... >> %LOGFILE%
call bundler -v >nul 2>&1
set BUNDLER_CHECK=!ERRORLEVEL!
if !BUNDLER_CHECK! EQU 0 (
    echo Bundler is already installed.
    echo Bundler is already installed. >> %LOGFILE%
    call bundler -v >> %LOGFILE%
) else (
    echo Installing Bundler...
    echo Installing Bundler... >> %LOGFILE%
    call gem install bundler >> %LOGFILE% 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo Failed to install Bundler.
        echo Failed to install Bundler. >> %LOGFILE%
        pause
        exit /b 1
    )
)

:: Step 4: Install bundle inside exploring-ai-backend repo
echo Installing bundle inside exploring-ai-backend repo...
echo Installing bundle inside exploring-ai-backend repo... >> %LOGFILE%
cd ..
if !ERRORLEVEL! NEQ 0 (
    echo Failed to change directory to exploring-ai-backend.
    echo Failed to change directory to exploring-ai-backend. >> %LOGFILE%
    pause
    exit /b 1
)

call bundle install >> %LOGFILE% 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo Failed to install bundle.
    echo Failed to install bundle. >> %LOGFILE%
    pause
    exit /b 1
)

echo Environment setup complete for Windows.
echo Environment setup complete for Windows. >> %LOGFILE%
echo Check %LOGFILE% for detailed log.
pause
endlocal