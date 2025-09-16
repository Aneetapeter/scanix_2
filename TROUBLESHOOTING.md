# Scanix Troubleshooting Guide

## Network Connection Issues

### Problem: "Cannot connect to backend server"

This error occurs when the Flutter app cannot reach the Python Flask backend.

### Solutions:

#### 1. Check if Backend is Running
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

You should see:
```
Starting Scanix Backend Server...
==================================================
Loading AI model...
✓ Model loaded successfully
==================================================
Server starting on http://localhost:5000
```

#### 2. Test Backend Connection
Open your browser and go to: http://localhost:5000/health

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

#### 3. Check Port Availability
Make sure port 5000 is not being used by another application:
```bash
# Windows
netstat -ano | findstr :5000

# macOS/Linux
lsof -i :5000
```

#### 4. Firewall Issues
- **Windows**: Allow Python through Windows Firewall
- **macOS**: Check System Preferences > Security & Privacy > Firewall
- **Linux**: Check `ufw` or `iptables` settings

#### 5. CORS Issues (Web Only)
If running Flutter web, the browser may block localhost connections. Try:
- Use `flutter run -d chrome --web-port 3000`
- Or use `127.0.0.1:5000` instead of `localhost:5000`

### Alternative: Demo Mode

If you can't get the backend running, the app will show demo data for doctors but image analysis won't work.

## Common Error Messages

### "Failed to analyze image"
- Backend server is not running
- Image file is corrupted or unsupported
- Network connection issues

### "Cannot connect to backend server"
- Backend server is not running
- Wrong URL configuration
- Firewall blocking connection
- Port already in use

### "Model failed to load"
- TensorFlow installation issues
- Insufficient memory
- Missing dependencies

## Quick Fixes

### 1. Restart Everything
```bash
# Stop all processes (Ctrl+C)
# Then restart:

# Terminal 1 - Backend
cd backend
python run.py

# Terminal 2 - Frontend
flutter run -d web-server --web-port 3000
```

### 2. Clear Flutter Cache
```bash
flutter clean
flutter pub get
flutter run
```

### 3. Reinstall Python Dependencies
```bash
cd backend
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### 4. Use Different Ports
If port 5000 is busy, modify `backend/run.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change to 5001
```

Then update `lib/services/api_service.dart`:
```dart
static String get baseUrl => 'http://localhost:5001';
```

## Platform-Specific Issues

### Windows
- Run Command Prompt as Administrator
- Check Windows Defender settings
- Ensure Python is in PATH

### macOS
- Use `python3` instead of `python`
- Check Xcode Command Line Tools: `xcode-select --install`
- Allow Python through Security & Privacy

### Linux
- Install Python development headers: `sudo apt-get install python3-dev`
- Check SELinux settings
- Ensure proper permissions

## Getting Help

If you're still having issues:

1. **Check the console output** for specific error messages
2. **Verify all dependencies** are installed correctly
3. **Test the backend** independently using curl or Postman
4. **Check network connectivity** between frontend and backend
5. **Review firewall and security settings**

### Test Backend with curl:
```bash
curl -X GET http://localhost:5000/health
curl -X GET http://localhost:5000/doctors
```

### Test with Postman:
- GET http://localhost:5000/health
- GET http://localhost:5000/doctors
- POST http://localhost:5000/analyze (with image file)

## Demo Mode

If you can't get the backend running, the app will still work with limited functionality:
- ✅ All pages and navigation work
- ✅ Doctor list shows demo data
- ✅ Contact form works (shows error but doesn't crash)
- ❌ Image analysis won't work
- ❌ Real AI detection unavailable

This is perfect for testing the UI and user experience while you set up the backend.
