@echo off
echo Starting AI Model Training...
echo.

cd /d "%~dp0"

python complete_training.py

echo.
echo Training process completed!
pause
