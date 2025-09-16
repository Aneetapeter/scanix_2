# Scanix Quick Start Guide

## 🚀 Get Running in 2 Minutes

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
# Double-click or run:
start_dev.bat
```

**macOS/Linux:**
```bash
chmod +x start_dev.sh
./start_dev.sh
```

### Option 2: Manual Setup

**Step 1: Start Backend**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

**Step 2: Start Frontend** (in new terminal)
```bash
flutter pub get
flutter run -d web-server --web-port 3000
```

## 🌐 Access Your App

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 🔧 Troubleshooting

### "Cannot connect to backend server"
1. Make sure backend is running: `python check_backend.py`
2. Check if port 5000 is free
3. Try different browser or incognito mode

### "Demo Mode" messages
- Backend is not running or not accessible
- App will still work with simulated AI analysis
- Perfect for testing UI without backend setup

### Backend won't start
1. Install Python dependencies: `pip install -r requirements.txt`
2. Check Python version: `python --version` (need 3.8+)
3. Try different port in `backend/run.py`

## 📱 What You Can Do

### With Backend Running:
- ✅ Real AI image analysis
- ✅ Upload images from gallery/camera
- ✅ Get confidence scores and recommendations
- ✅ Connect with doctors
- ✅ All features work

### In Demo Mode:
- ✅ Upload images (simulated analysis)
- ✅ Browse all pages
- ✅ Test UI and navigation
- ✅ See doctor listings
- ❌ No real AI analysis

## 🎯 Next Steps

1. **Test the app** - Upload an image and see the analysis
2. **Explore pages** - Check out About, Doctors, How It Works
3. **Read documentation** - See README.md for full details
4. **Customize** - Modify colors, add your own content

## 🆘 Need Help?

- **Full Guide**: See README.md
- **Troubleshooting**: See TROUBLESHOOTING.md
- **Installation**: See INSTALLATION.md

---

**Pro Tip**: If you just want to see the UI, you can run the Flutter app without the backend - it will work in demo mode!
