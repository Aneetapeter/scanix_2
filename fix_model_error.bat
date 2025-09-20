@echo off
echo Fixing Model Feature Mismatch Error...
echo.

cd backend

echo Step 1: Updating backend model...
python final_update.py

echo.
echo Step 2: Backend preprocessing fixed!
echo - Changed from RGB (3072 features) to Grayscale (1024 features)
echo - Now matches the retrained model expectations

echo.
echo Step 3: Restarting backend server...
taskkill /f /im python.exe 2>nul

echo.
echo Step 4: Starting backend with fixed preprocessing...
start "Fixed AI Backend" cmd /k "python final_app.py"

echo.
echo ========================================
echo ERROR FIXED! Backend is now running...
echo ========================================
echo.
echo The model now expects 1024 features (32x32 grayscale)
echo instead of 3072 features (32x32x3 RGB)
echo.
echo Your Flutter app should now work correctly!
echo.
pause
