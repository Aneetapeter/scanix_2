@echo off
echo Starting Both Flutter App and AI Server...
echo.

echo Starting AI Server...
start "AI Server" cmd /k "cd /d %~dp0 && python final_app.py"

echo Waiting 5 seconds for AI server to start...
timeout /t 5 /nobreak >nul

echo Starting Flutter App...
cd /d "%~dp0\.."
flutter run

pause
