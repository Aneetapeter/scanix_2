@echo off
echo Starting Both Flutter App and AI Backend...
echo.

echo Starting AI Backend Server in background...
start "AI Backend" cmd /k "cd backend && python app_with_ai.py"

echo Waiting 5 seconds for AI server to start...
timeout /t 5 /nobreak >nul

echo Starting Flutter App...
flutter run

pause
