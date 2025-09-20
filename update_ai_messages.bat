@echo off
echo 🔄 UPDATING AI RESPONSE MESSAGES
echo ================================
echo.
echo This will update the AI to show clear messages:
echo - "Normal." for normal faces
echo - "Paralysis detected" for paralyzed faces
echo.

echo Step 1: Stopping old backend...
echo ==============================
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Starting updated backend...
echo ==================================
cd backend
start "Updated AI Backend" cmd /k "python final_app.py"

echo.
echo Step 3: Waiting for backend to start...
echo ======================================
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo ✅ AI MESSAGES UPDATED!
echo ========================================
echo.
echo Your AI now shows:
echo ✅ "Normal." for normal faces
echo ✅ "Paralysis detected" for paralyzed faces
echo ✅ Clear and direct responses
echo.
echo Test your app at: http://localhost:3002
echo.
pause
