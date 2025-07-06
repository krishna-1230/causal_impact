@echo off
echo ===================================================
echo    Causal Impact Engine - Marketing Attribution
echo ===================================================
echo.

REM Check if Anaconda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Anaconda/Miniconda is not installed or not in PATH.
    echo Please install Anaconda or Miniconda from https://www.anaconda.com/download/
    echo.
    pause
    exit /b 1
)

echo Activating Anaconda environment: causal_impact
echo.

REM Try to activate the environment
CALL conda activate causal_impact 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Environment "causal_impact" not found. Creating it now...
    echo This may take a few minutes...
    echo.
    
    REM Create the environment from the environment.yml file
    if exist environment.yml (
        CALL conda env create -f environment.yml
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to create environment.
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: environment.yml file not found.
        echo Please run this script from the project root directory.
        pause
        exit /b 1
    )
    
    CALL conda activate causal_impact
)

echo.
echo Changing directory to causal_impact_engine...
cd causal_impact_engine

echo.
echo Running Streamlit app: frontend/app.py
echo.
echo NOTE: If the app doesn't start, please ensure all dependencies are installed:
echo       pip install -r requirements.txt
echo.

streamlit run frontend/app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to run Streamlit app.
    echo Please check that all dependencies are installed correctly.
)

echo.
echo Streamlit app has exited.
pause
