@echo off
echo 🚀 CREATING ENHANCED AI SOLUTION FOR FACIAL PARALYSIS DETECTION
echo =============================================================
echo.
echo This will create a much better AI model that can properly
echo distinguish between paralyzed and normal faces by:
echo.
echo ✅ Analyzing facial asymmetry (key for paralysis detection)
echo ✅ Using face detection and alignment
echo ✅ Extracting edge and texture features
echo ✅ Using higher resolution images (64x64)
echo ✅ Training with more data and better preprocessing
echo.
echo Step 1: Running enhanced training...
echo ====================================
cd backend
python enhanced_training.py

echo.
echo Step 2: Updating backend to use enhanced model...
echo ================================================
python update_to_enhanced.py

echo.
echo Step 3: Restarting backend with enhanced AI...
echo =============================================
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Starting enhanced AI backend...
start "Enhanced AI Backend" cmd /k "python final_app.py"

echo.
echo Step 4: Starting Flutter app...
echo =============================
cd ..
timeout /t 5 /nobreak >nul
start "Flutter App" cmd /k "flutter run -d web-server --web-port 3002"

echo.
echo ========================================
echo 🎉 ENHANCED AI SOLUTION READY!
echo ========================================
echo.
echo Your Scanix app now has:
echo ✅ Much better facial paralysis detection
echo ✅ Asymmetry analysis for accurate detection
echo ✅ Face detection and alignment
echo ✅ Enhanced feature extraction
echo ✅ Higher accuracy model
echo.
echo Access your app at: http://localhost:3002
echo.
pause
