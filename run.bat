@echo off

set PYTHON_VER=3.10.11

:: Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

IF NOT EXIST venv (
    echo Error: Virtual environment venv not found. Please run skeleton.bat first to create the environment and install required packages.
    exit /b 1
)

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Activate the virtual environment
call .\venv\Scripts\activate.bat

echo Running app.py within the virtual environment...
python app.py --image-folder ./images --beam-search-numbers 2