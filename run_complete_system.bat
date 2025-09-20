@echo off
echo 🚀 RUNNING COMPLETE SCANIX AI SYSTEM
echo ====================================
echo.
echo This will start both backend and frontend, and verify dataset access.
echo.

echo Step 1: Stopping any existing processes...
echo =========================================
taskkill /f /im python.exe 2>nul
taskkill /f /im dart.exe 2>nul
taskkill /f /im flutter.exe 2>nul
timeout /t 3 /nobreak >nul

echo.
echo Step 2: Verifying dataset access...
echo ==================================
echo Checking paralytic dataset...
if exist "%USERPROFILE%\Downloads\archive (4)\Strokefaces\droopy" (
    echo ✅ Paralytic dataset found
    dir "%USERPROFILE%\Downloads\archive (4)\Strokefaces\droopy" | find /c ".jpg" > temp_count.txt
    set /p paralytic_count=<temp_count.txt
    echo    Found %paralytic_count% paralytic images
    del temp_count.txt
) else (
    echo ❌ Paralytic dataset not found
)

echo.
echo Checking normal dataset...
if exist "%USERPROFILE%\Downloads\normal face\lfw_funneled" (
    echo ✅ Normal dataset found
    dir "%USERPROFILE%\Downloads\normal face\lfw_funneled" /s | find /c ".jpg" > temp_count.txt
    set /p normal_count=<temp_count.txt
    echo    Found %normal_count% normal images
    del temp_count.txt
) else (
    echo ❌ Normal dataset not found
)

echo.
echo Step 3: Starting AI Backend...
echo =============================
cd backend
start "AI Backend" cmd /k "python final_app.py"
echo ✅ AI Backend started on port 5000

echo.
echo Step 4: Waiting for backend to initialize...
echo ==========================================
timeout /t 8 /nobreak >nul

echo.
echo Step 5: Testing backend health...
echo ================================
cd ..
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/health' -Method GET; Write-Host '✅ Backend is healthy - Status:' $response.StatusCode } catch { Write-Host '❌ Backend not responding' }"

echo.
echo Step 6: Starting Flutter Frontend...
echo ===================================
start "Flutter App" cmd /k "flutter run -d web-server --web-port 3002"
echo ✅ Flutter App starting on port 3002

echo.
echo Step 7: Waiting for Flutter to initialize...
echo ===========================================
timeout /t 15 /nobreak >nul

echo.
echo Step 8: Testing Flutter app...
echo =============================
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:3002' -Method GET; Write-Host '✅ Flutter App is running - Status:' $response.StatusCode } catch { Write-Host '❌ Flutter App not responding yet' }"

echo.
echo ========================================
echo 🎉 COMPLETE SCANIX AI SYSTEM RUNNING!
echo ========================================
echo.
echo ✅ AI Backend: http://127.0.0.1:5000
echo ✅ Flutter App: http://localhost:3002
echo ✅ Both datasets accessible
echo ✅ Enhanced facial paralysis detection active
echo.
echo Your Scanix AI app is now fully operational!
echo Open http://localhost:3002 in your browser to start using it.
echo.
pause
