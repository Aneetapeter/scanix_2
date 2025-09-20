# PowerShell script to start both Flutter app and AI backend

Write-Host "🚀 Starting Flutter App + AI Backend" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Start AI backend in background
Write-Host "Starting AI Backend Server..." -ForegroundColor Yellow
$aiJob = Start-Job -ScriptBlock {
    Set-Location "backend"
    python app_with_ai.py
}

# Wait a moment for AI server to start
Write-Host "Waiting for AI server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if AI server is running
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ AI Backend is running!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ AI Backend might not be ready yet" -ForegroundColor Yellow
}

# Start Flutter app
Write-Host "Starting Flutter App..." -ForegroundColor Yellow
flutter run

# Clean up background job when done
Write-Host "Stopping AI Backend..." -ForegroundColor Yellow
Stop-Job $aiJob
Remove-Job $aiJob
