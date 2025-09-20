@echo off
echo 🚀 FULL RESTART - SCANIX AI APP
echo =================================
echo.
echo This script will stop all running Python and Flutter processes,
echo then restart your enhanced AI backend and Flutter web app.
echo.

echo Step 1: Stopping all Python and Flutter processes...
echo ===================================================
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM dart.exe /T >nul 2>&1
taskkill /F /IM flutter.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
echo ✅ All relevant processes stopped.

echo.
echo Step 2: Starting enhanced AI Backend...
echo =====================================
cd backend
start "Enhanced AI Backend" cmd /k "python final_app.py"
echo Waiting 5 seconds for AI backend to start...
timeout /t 5 /nobreak >nul
cd ..

echo.
echo Step 3: Starting Flutter Web App on port 3002...
echo ===============================================
start "Flutter App" cmd /k "flutter run -d web-server --web-port 3002"
echo Waiting 10 seconds for Flutter app to start...
timeout /t 10 /nobreak >nul

echo.
echo ========================================
echo 🎉 SCANIX AI APP FULLY RESTARTED!
echo ========================================
echo.
echo Access your Flutter app at: http://localhost:3002
echo.
pause
