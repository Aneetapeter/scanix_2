@echo off
echo Starting Scanix AI System...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "cd backend && python app.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting Flutter App...
start "Flutter" cmd /k "flutter run"

echo.
echo Scanix AI System is starting...
echo Backend: http://localhost:5000
echo Frontend: Will open automatically
echo.
pause
