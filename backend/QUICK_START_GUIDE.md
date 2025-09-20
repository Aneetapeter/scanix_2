# Face Classification Quick Start Guide

## 🚀 How to Run Both Training and API

### **Method 1: Automated Script (Recommended)**

**Windows:**
```bash
cd backend
run_training_and_api.bat
```

**Linux/Mac:**
```bash
cd backend
chmod +x run_training_and_api.sh
./run_training_and_api.sh
```

### **Method 2: Manual Steps**

#### **Step 1: Train the Model**
```bash
cd backend
python train_new_face_model.py
```
**Expected output:**
- Loading 1000 paralyzed images
- Loading 1000 normal images  
- Training Random Forest, SVM, and Logistic Regression
- Creating evaluation plots
- Saving model to `models_new/` folder

#### **Step 2: Start the API Server**
```bash
cd backend
python face_classification_api.py
```
**Expected output:**
- API server starts on http://localhost:5001
- Endpoints available:
  - `GET /health` - Health check
  - `POST /classify` - Classify face images
  - `GET /model_info` - Get model information

### **Method 3: Run in Separate Terminals**

**Terminal 1 (Training):**
```bash
cd backend
python train_new_face_model.py
```

**Terminal 2 (API - after training completes):**
```bash
cd backend
python face_classification_api.py
```

## 📊 What Happens During Training

1. **Data Loading**: Loads 1000 paralyzed + 1000 normal images
2. **Feature Extraction**: Extracts 2000+ features per image
3. **Model Training**: Trains 3 different models
4. **Evaluation**: Tests performance and creates plots
5. **Model Saving**: Saves best model to `models_new/`

## 🔧 Using the API

### **Test with curl:**
```bash
curl -X POST http://localhost:5001/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

### **Test with Python:**
```python
import requests
import base64

# Load and encode image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post("http://localhost:5001/classify", 
                        json={"image": image_data})
result = response.json()
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📁 Output Files

After training, you'll find:
- `models_new/face_classification_model.pkl` - Trained model
- `models_new/face_scaler.pkl` - Feature scaler
- `models_new/model_info.json` - Model metadata
- `models_new/evaluation_results.json` - Performance metrics
- `models_new/plots/` - Visualization plots

## 🧪 Test Individual Images

```bash
cd backend
python predict_face.py "path/to/your/image.jpg"
```

## ⚠️ Troubleshooting

**If training fails:**
- Check that image paths are correct
- Ensure you have enough disk space
- Check Python dependencies are installed

**If API fails to start:**
- Make sure training completed successfully
- Check that model files exist in `models_new/`
- Ensure port 5001 is available

**If predictions are inaccurate:**
- Try with higher quality images
- Ensure faces are clearly visible
- Check that images are properly lit
