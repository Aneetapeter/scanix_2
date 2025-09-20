@echo off
echo Starting Scanix AI Application...

REM Start Backend
echo Starting backend server...
start cmd /k "cd backend && python improved_main.py"

REM Wait a moment for the backend to initialize
timeout /t 5 /nobreak > NUL

REM Start Frontend
echo Starting Flutter frontend...
start cmd /k "flutter run"

echo Scanix AI Application started.
echo.
echo Backend: http://localhost:5000
echo Frontend: Will open in a new window
echo.
echo Press any key to close this window...
pause > NUL