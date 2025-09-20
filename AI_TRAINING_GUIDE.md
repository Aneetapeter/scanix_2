# AI Facial Paralysis Detection System

This document provides a comprehensive guide for the AI-powered facial paralysis detection system that has been trained to distinguish between normal faces and faces with paralysis.

## 🎯 System Overview

The system uses machine learning to analyze facial images and classify them into two categories:
- **Label 0**: Normal Face (no paralysis)
- **Label 1**: Paralyzed Face (droopy)

## 📊 Dataset Information

### Training Data
- **Normal Faces**: 226 images from LFW (Labeled Faces in the Wild) dataset
- **Paralyzed Faces**: 621 images from medical sources showing facial paralysis
- **Total Training Images**: 847 images
- **Validation Images**: 174 images
- **Test Images**: 200 images

### Data Preprocessing
- Images resized to 128x128 pixels for traditional ML features
- Images resized to 224x224 pixels for CNN features
- Grayscale conversion for asymmetry analysis
- Data augmentation applied during CNN training

## 🤖 Model Architecture

### Current Model: Random Forest Classifier
- **Algorithm**: Random Forest with 500 estimators
- **Features**: 4,360 engineered features
- **Accuracy**: 88.5% on validation set
- **Key Features**:
  - Facial asymmetry analysis (most important)
  - Eye region asymmetry detection
  - Mouth region asymmetry detection
  - Edge and texture asymmetry
  - Histogram comparison
  - Regional asymmetry analysis

### Feature Engineering
The system extracts comprehensive features specifically designed for paralysis detection:

1. **Facial Asymmetry Features** (Most Critical)
   - Left vs right half comparison
   - Multiple asymmetry metrics (mean, std, max, median)

2. **Eye Region Analysis**
   - Asymmetry in eye regions
   - Critical for detecting drooping

3. **Mouth Region Analysis**
   - Asymmetry in mouth regions
   - Important for facial expression analysis

4. **Edge and Texture Analysis**
   - Gradient-based asymmetry detection
   - Texture pattern comparison

5. **Statistical Features**
   - Brightness, contrast, and distribution analysis
   - Skewness and kurtosis measurements

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Flutter SDK
- Required Python packages (see requirements.txt)

### Installation

1. **Backend Setup**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Flutter Setup**:
   ```bash
   flutter pub get
   ```

### Running the System

1. **Start the AI Service**:
   ```bash
   cd backend
   python simple_inference_service.py
   ```
   The service will start on `http://localhost:5000`

2. **Run the Flutter App**:
   ```bash
   flutter run
   ```

## 📡 API Endpoints

### Health Check
```
GET /health
```
Returns service status and model information.

### Predict from Base64 Image
```
POST /predict
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

### Predict from File Upload
```
POST /predict_file
Content-Type: multipart/form-data

file: [image_file]
```

### Get Model Information
```
GET /model_info
```

## 🔧 Training New Models

### 1. Enhanced CNN Training
```bash
cd backend
python enhanced_cnn_trainer.py
```

### 2. Hybrid Training (CNN + Traditional ML)
```bash
cd backend
python hybrid_trainer.py
```

### 3. Improved Simple Training
```bash
cd backend
python improved_simple_trainer.py
```

## 📈 Model Performance

### Current Model Results
- **Accuracy**: 88.5%
- **Precision**: 96% (Paralysis), 73% (Normal)
- **Recall**: 87% (Paralysis), 92% (Normal)
- **F1-Score**: 91% (Paralysis), 81% (Normal)

### Confusion Matrix
```
                 Predicted
                Normal  Paralysis
Actual Normal      44         4
       Paralysis   16       110
```

## 🎯 Usage in Flutter App

The Flutter app automatically integrates with the AI service:

1. **Image Capture**: Users can take photos or select from gallery
2. **AI Analysis**: Images are sent to the AI service for analysis
3. **Results Display**: Results show:
   - Prediction (Normal/Paralysis)
   - Confidence level
   - Detailed recommendations
   - Model information

### Example Integration
```dart
// Analyze image using AI
final result = await AIAnalysisService.analyzeImage(imageFile);

// Display results
print('Prediction: ${result.hasParalysis ? "Paralysis" : "Normal"}');
print('Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%');
print('Recommendation: ${result.recommendation}');
```

## 🔍 Model Interpretability

### Feature Importance
The most important features for paralysis detection are:
1. Facial asymmetry metrics (highest importance)
2. Eye region asymmetry
3. Mouth region asymmetry
4. Edge asymmetry patterns
5. Texture asymmetry

### Confidence Levels
- **High Confidence (>80%)**: Strong evidence for prediction
- **Moderate Confidence (60-80%)**: Moderate evidence
- **Low Confidence (<60%)**: Weak evidence, consider retaking photo

## 🚨 Important Notes

### Medical Disclaimer
- This system is for educational and research purposes only
- Not a substitute for professional medical diagnosis
- Always consult healthcare professionals for medical concerns
- Results should be interpreted by qualified medical personnel

### Limitations
- Model accuracy is 88.5% - not 100% reliable
- Performance may vary with image quality
- Lighting and angle can affect results
- Not suitable for clinical diagnosis

### Recommendations
- Use good lighting for photos
- Ensure face is clearly visible
- Take multiple angles if uncertain
- Always consult medical professionals

## 🔧 Troubleshooting

### Common Issues

1. **AI Service Not Starting**:
   - Check if Python dependencies are installed
   - Ensure port 5000 is available
   - Check model files exist in `backend/models/`

2. **Flutter App Connection Issues**:
   - Verify AI service is running on localhost:5000
   - Check network connectivity
   - App will fall back to local analysis if AI service unavailable

3. **Model Loading Errors**:
   - Ensure model files are present:
     - `ai_model.pkl`
     - `scaler.pkl`
     - `model_info.json`

## 📚 Technical Details

### Model Training Process
1. **Data Loading**: Load and preprocess images
2. **Feature Extraction**: Extract 4,360 engineered features
3. **Data Splitting**: 80% train, 20% test
4. **Model Training**: Random Forest with class balancing
5. **Validation**: Cross-validation and performance metrics
6. **Model Saving**: Save trained model and scaler

### Feature Engineering Details
- **Asymmetry Analysis**: Compare left and right halves of face
- **Regional Analysis**: Analyze specific facial regions
- **Statistical Analysis**: Extract statistical properties
- **Edge Detection**: Analyze facial contours and edges
- **Texture Analysis**: Compare texture patterns

## 🎉 Success Metrics

The system successfully:
- ✅ Achieves 88.5% accuracy on validation data
- ✅ Provides detailed confidence scores
- ✅ Offers comprehensive recommendations
- ✅ Integrates seamlessly with Flutter app
- ✅ Falls back gracefully if AI service unavailable
- ✅ Handles various image formats and qualities

## 🔮 Future Improvements

Potential enhancements:
- Increase dataset size for better accuracy
- Implement ensemble methods
- Add real-time video analysis
- Improve feature engineering
- Add more sophisticated CNN architectures
- Implement transfer learning
- Add multi-class classification (different types of paralysis)

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the model performance metrics
- Ensure all dependencies are properly installed
- Verify the AI service is running correctly

---

**Remember**: This is a research and educational tool. Always consult qualified medical professionals for any health concerns.
