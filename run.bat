@echo off
:: Document Similarity Tool - Windows Batch Script
:: This script provides easy access to the document similarity system

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Document Similarity Detection System
echo ========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Show menu
:menu
echo Please choose an option:
echo.
echo 1. Setup and verify system
echo 2. Run simple example
echo 3. Run full demo
echo 4. Process documents in sample_pdfs
echo 5. Search for similar documents
echo 6. Find duplicates
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto example
if "%choice%"=="3" goto demo
if "%choice%"=="4" goto process
if "%choice%"=="5" goto search
if "%choice%"=="6" goto duplicates
if "%choice%"=="7" goto exit
echo Invalid choice. Please try again.
goto menu

:setup
echo.
echo Running setup and verification...
python setup.py
pause
goto menu

:example
echo.
echo Running simple example...
python example.py
pause
goto menu

:demo
echo.
echo Running full demonstration...
python main.py demo
pause
goto menu

:process
echo.
echo Processing documents in data/sample_pdfs...
python main.py process -d data/sample_pdfs
pause
goto menu

:search
echo.
set /p query="Enter search query: "
echo Searching for: !query!
python main.py search -q "!query!" -k 5
pause
goto menu

:duplicates
echo.
echo Finding duplicate documents...
python main.py duplicates -t 0.9
pause
goto menu

:exit
echo.
echo Thank you for using Document Similarity Detection System!
pause
exit /b 0
