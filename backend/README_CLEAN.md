# 🎯 **Clean AI Facial Paralysis Detection System**

## ✅ **Essential Files Only**

### **Core AI System:**
- `simple_app.py` - Main Flask API server with trained AI
- `quick_train.py` - AI training script (already completed)
- `predict_facial_paralysis.py` - Standalone prediction script

### **Trained AI Model:**
- `models/ai_model.pkl` - Your trained AI model (94.12% accuracy)
- `models/model_info.json` - Model information and metrics

### **Startup Scripts:**
- `start_ai_server.bat` - Start AI server
- `start_flutter.bat` - Start Flutter app
- `start_both_apps.bat` - Start both at once

### **Dataset (Organized):**
- `data/train/` - Training images (226 normal + 716 paralysis)
- `data/validation/` - Validation images (48 normal + 153 paralysis)
- `data/test/` - Test images (50 normal + 155 paralysis)
- `data/raw_data/` - Original datasets (LFW + Stroke faces)

### **Configuration:**
- `requirements.txt` - Python dependencies
- `app.py` - Original Flask app (backup)

## 🚀 **How to Use:**

### **Start Everything:**
1. Double-click `start_both_apps.bat`
2. Your Flutter app will connect to the AI backend
3. Start analyzing images for facial paralysis!

### **Manual Start:**
```bash
# Terminal 1 - AI Server
python simple_app.py

# Terminal 2 - Flutter App
flutter run
```

## 📊 **AI Performance:**
- **Accuracy:** 94.12%
- **Training Images:** 847 total
- **Model Type:** Random Forest Classifier
- **API Endpoint:** http://127.0.0.1:5000

## 🎉 **You're Ready!**

Your clean, optimized AI system is ready for facial paralysis detection!
