@echo off
echo Starting Scanix Development Environment
echo ======================================

echo.
echo Checking if backend is already running...
python check_backend.py
if %errorlevel% neq 0 (
    echo.
    echo Starting Python Flask Backend...
    start "Backend Server" cmd /k "cd backend && python run.py"
    
    echo.
    echo Waiting for backend to start...
    timeout /t 8 /nobreak > nul
    
    echo.
    echo Checking backend status...
    python check_backend.py
    if %errorlevel% neq 0 (
        echo.
        echo ⚠️  Backend may not be ready yet. The app will run in demo mode.
    )
) else (
    echo.
    echo ✅ Backend is already running!
)

echo.
echo Starting Flutter Frontend...
start "Flutter App" cmd /k "flutter run -d web-server --web-port 3000"

echo.
echo Development environment started!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo If you see "Demo Mode" messages, the backend is not running.
echo Check the Backend Server window for any errors.
echo.
echo Press any key to exit...
pause > nul
