Write-Host "🚀 Starting Scanix AI Facial Paralysis Detection App..." -ForegroundColor Green
Write-Host ""

# Start AI Backend Server
Write-Host "📡 Starting AI Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; python final_app.py"

# Wait for backend to start
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Start Flutter App
Write-Host "📱 Starting Flutter App..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; flutter run -d web-server --web-port 3001"

Write-Host ""
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host "🌐 Flutter App: http://localhost:3001" -ForegroundColor Cyan
Write-Host "🔬 AI Backend: http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
