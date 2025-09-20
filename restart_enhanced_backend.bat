@echo off
echo 🚀 RESTARTING ENHANCED AI BACKEND
echo =================================
echo.
echo This will restart your backend with enhanced facial paralysis detection
echo that properly analyzes facial asymmetry and texture features.
echo.

echo Step 1: Stopping old backend...
echo ==============================
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Starting enhanced backend...
echo ===================================
cd backend
start "Enhanced AI Backend" cmd /k "python final_app.py"

echo.
echo Step 3: Starting Flutter app...
echo ==============================
cd ..
timeout /t 5 /nobreak >nul
start "Flutter App" cmd /k "flutter run -d web-server --web-port 3002"

echo.
echo ========================================
echo 🎉 ENHANCED AI BACKEND RESTARTED!
echo ========================================
echo.
echo Your Scanix app now has:
echo ✅ Enhanced facial asymmetry analysis
echo ✅ Texture and edge feature extraction
echo ✅ Better paralysis detection accuracy
echo ✅ No OpenCV dependency issues
echo.
echo Access your app at: http://localhost:3002
echo.
pause
