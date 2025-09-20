@echo off
echo Updating Backend with Retrained Model...
echo.

cd /d "%~dp0"

python update_model.py

echo.
echo Backend update completed!
echo Please restart your backend server to use the new model.
pause
