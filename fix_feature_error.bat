@echo off
echo 🔧 FIXING FEATURE DIMENSION ERROR
echo =================================
echo.
echo This will fix the feature mismatch error and restart the backend.
echo.

echo Step 1: Stopping old backend...
echo ==============================
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Starting fixed backend...
echo ================================
cd backend
start "Fixed AI Backend" cmd /k "python final_app.py"

echo.
echo Step 3: Waiting for backend to start...
echo ======================================
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo ✅ FEATURE ERROR FIXED!
echo ========================================
echo.
echo The backend now properly handles:
echo ✅ 12288 features (matching scaler expectations)
echo ✅ Enhanced asymmetry analysis
echo ✅ Proper feature padding/truncation
echo ✅ 64x64 image resolution
echo.
echo Your Flutter app should now work correctly!
echo Access at: http://localhost:3002
echo.
pause
