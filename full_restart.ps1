Write-Host "🚀 FULL RESTART - SCANIX AI APP" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Stopping all Python and Flutter processes..." -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Yellow

# Stop all Python processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "✅ Python processes stopped" -ForegroundColor Green

# Stop all Dart processes
Get-Process | Where-Object {$_.ProcessName -like "*dart*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "✅ Dart processes stopped" -ForegroundColor Green

# Stop all Flutter processes
Get-Process | Where-Object {$_.ProcessName -like "*flutter*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "✅ Flutter processes stopped" -ForegroundColor Green

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "Step 2: Starting enhanced AI Backend..." -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow

Set-Location "backend"
Start-Process -FilePath "python" -ArgumentList "final_app.py" -WindowStyle Normal
Write-Host "✅ Enhanced AI Backend started" -ForegroundColor Green

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "Step 3: Starting Flutter Web App on port 3002..." -ForegroundColor Yellow
Write-Host "==============================================" -ForegroundColor Yellow

Set-Location ".."
Start-Process -FilePath "flutter" -ArgumentList "run", "-d", "web-server", "--web-port", "3002" -WindowStyle Normal
Write-Host "✅ Flutter App started" -ForegroundColor Green

Start-Sleep -Seconds 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "🎉 SCANIX AI APP FULLY RESTARTED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access your Flutter app at: http://localhost:3002" -ForegroundColor Cyan
Write-Host "AI Backend running at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
