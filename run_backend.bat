@echo off
echo Starting PSL Translator Backend...
echo.
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to start the backend.
    echo Please ensure Python is installed and all dependencies are met.
    pause
)
pause
