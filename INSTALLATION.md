# Scanix Installation Guide

## Quick Start

### Option 1: Automated Setup (Windows)
1. Run `start_dev.bat` to start both backend and frontend automatically
2. Backend will start on http://localhost:5000
3. Frontend will start on http://localhost:3000

### Option 2: Manual Setup

## Prerequisites

- **Flutter SDK** 3.9.2 or higher
- **Python** 3.8 or higher
- **Git** (for cloning)

## Step 1: Clone Repository
```bash
git clone <repository-url>
cd scanix_2
```

## Step 2: Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start backend server**:
   ```bash
   python run.py
   ```
   
   Backend will be available at: http://localhost:5000

## Step 3: Frontend Setup

1. **Return to project root**:
   ```bash
   cd ..
   ```

2. **Install Flutter dependencies**:
   ```bash
   flutter pub get
   ```

3. **Run Flutter app**:
   ```bash
   # For web
   flutter run -d web-server --web-port 3000
   
   # For desktop
   flutter run -d windows
   flutter run -d macos
   flutter run -d linux
   
   # For mobile
   flutter run
   ```

## Step 4: Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## Troubleshooting

### Common Issues

1. **Flutter dependencies not found**:
   ```bash
   flutter clean
   flutter pub get
   ```

2. **Python dependencies issues**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Port already in use**:
   - Change ports in `backend/run.py` and `flutter run` command
   - Update `baseUrl` in `lib/services/api_service.dart`

4. **CORS errors**:
   - Ensure backend is running before frontend
   - Check CORS configuration in `backend/app.py`

### Platform-Specific Issues

#### Windows
- Use `start_dev.bat` for easy startup
- Ensure Python and Flutter are in PATH

#### macOS/Linux
- Use `start_dev.sh` for easy startup
- Make script executable: `chmod +x start_dev.sh`

## Development Mode

### Hot Reload
- **Flutter**: Changes automatically reload
- **Backend**: Restart server after changes

### Debugging
- **Flutter**: Use VS Code or Android Studio debugger
- **Backend**: Use Python debugger or print statements

## Production Deployment

### Frontend
```bash
flutter build web
# Deploy build/web/ directory to hosting service
```

### Backend
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Configuration

### Environment Variables
Create `.env` file in backend directory:
```
FLASK_ENV=production
MODEL_PATH=path/to/model
DATABASE_URL=your_database_url
```

### API Configuration
Update `baseUrl` in `lib/services/api_service.dart`:
```dart
static const String baseUrl = 'https://your-api-domain.com';
```

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review error logs in terminal
3. Ensure all dependencies are installed
4. Verify ports are available
5. Contact support: support@scanix.ai

## Next Steps

After successful installation:
1. Test image upload functionality
2. Verify AI analysis works
3. Check doctor integration
4. Test all navigation pages
5. Review FAQ and disclaimers

---

**Note**: This is a development setup. For production deployment, additional security and performance configurations are required.
