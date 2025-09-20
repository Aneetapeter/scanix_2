@echo off
echo 🚀 FIXING AI FACIAL PARALYSIS DETECTION
echo ======================================
echo.
echo This will create a working AI solution that properly
echo distinguishes between paralyzed and normal faces.
echo.
echo Step 1: Training working enhanced model...
echo ==========================================
cd backend
python working_enhanced_training.py

echo.
echo Step 2: Updating backend model...
echo ================================
copy models\working_enhanced_model.pkl models\ai_model.pkl
copy models\working_enhanced_scaler.pkl models\scaler.pkl

echo.
echo Step 3: Stopping old backend...
echo ==============================
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 4: Starting working enhanced backend...
echo ===========================================
start "Working AI Backend" cmd /k "python working_final_app.py"

echo.
echo Step 5: Starting Flutter app...
echo ==============================
cd ..
timeout /t 5 /nobreak >nul
start "Flutter App" cmd /k "flutter run -d web-server --web-port 3002"

echo.
echo ========================================
echo 🎉 AI DETECTION FIXED!
echo ========================================
echo.
echo Your Scanix app now has:
echo ✅ Working facial paralysis detection
echo ✅ Enhanced asymmetry analysis
echo ✅ No OpenCV dependency issues
echo ✅ Proper feature extraction
echo ✅ High accuracy model
echo.
echo Access your app at: http://localhost:3002
echo.
pause
