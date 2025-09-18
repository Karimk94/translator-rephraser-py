@echo off
REM ============================================================================
REM  Run script for the Local Image to Text API
REM
REM  This script will activate the virtual environment and start the Flask API.
REM ============================================================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting the Image to Text API server...
echo Press CTRL+C to stop the server.
echo.

REM Run the Flask application
python app.py

echo.
echo Server has been stopped.
pause
