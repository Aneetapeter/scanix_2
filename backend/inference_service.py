#!/usr/bin/env python3
"""
Exact Inference Service for Facial Paralysis Detection
Uses standardized preprocessing to ensure training/inference consistency
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from datetime import datetime
from standardized_preprocessing import preprocessor

class ExactFacialParalysisInference:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.ml_model = None
        self.scaler = None
        self.model_info = None
        self.image_size = (128, 128)  # EXACT same as training
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load ML model
            self.ml_model = joblib.load(self.models_dir / 'ai_model.pkl')
            self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
            
            # Load model info
            with open(self.models_dir / 'model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            print("✅ Models loaded successfully")
            print(f"   Model type: {self.model_info.get('model_type', 'Unknown')}")
            print(f"   Accuracy: {self.model_info.get('accuracy', 0.0):.4f}")
            print(f"   Expected features: {self.ml_model.n_features_in_}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def extract_paralysis_features(self, image):
        """Extract features using standardized preprocessing"""
        try:
            print(f"🔍 Starting standardized feature extraction...")
            print(f"   Original image size: {image.size}")
            print(f"   Original image mode: {image.mode}")
            
            # Use standardized preprocessing
            features = preprocessor.extract_paralysis_features(image)
            
            if features is None:
                print(f"❌ Feature extraction failed")
                return None
            
            print(f"✅ Standardized feature extraction completed")
            print(f"   Total features: {len(features)}")
            print(f"   Expected features: {self.ml_model.n_features_in_}")
            
            return features
            
        except Exception as e:
            print(f"❌ Error in standardized feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, image):
        """Predict facial paralysis from image using standardized preprocessing"""
        try:
            print(f"🚀 Starting prediction with standardized preprocessing...")
            
            # Extract features using standardized preprocessing
            features = self.extract_paralysis_features(image)
            if features is None:
                print(f"❌ Feature extraction failed")
                return None
            
            print(f"✅ Features extracted successfully")
            print(f"   Feature count: {len(features)}")
            print(f"   Feature shape: {features.shape}")
            print(f"   Feature dtype: {features.dtype}")
            
            # Check if feature count matches model expectations
            expected_features = self.ml_model.n_features_in_
            if len(features) != expected_features:
                print(f"❌ Feature count mismatch!")
                print(f"   Expected: {expected_features}")
                print(f"   Got: {len(features)}")
                return None
            
            # Scale features
            print(f"🔧 Scaling features...")
            features_scaled = self.scaler.transform([features])
            print(f"   Scaled features shape: {features_scaled.shape}")
            
            # Make prediction
            print(f"🤖 Making prediction...")
            prediction = self.ml_model.predict(features_scaled)[0]
            probability = self.ml_model.predict_proba(features_scaled)[0]
            
            print(f"✅ Prediction successful!")
            print(f"   Prediction: {prediction}")
            print(f"   Probabilities: {probability}")
            print(f"   Confidence: {max(probability):.3f}")
            
            return {
                'prediction': int(prediction),
                'confidence': float(max(probability)),
                'probabilities': {
                    'normal': float(probability[0]),
                    'paralysis': float(probability[1])
                },
                'label': 'Normal Face' if prediction == 0 else 'Paralyzed Face (Droopy)',
                'description': 'The face appears to be normal with no signs of paralysis.' if prediction == 0 else 'The face shows signs of facial paralysis or drooping.'
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            import traceback
            print(f"📋 Full traceback:")
            traceback.print_exc()
            return None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize inference service
inference_service = None

def initialize_service():
    """Initialize the inference service"""
    global inference_service
    try:
        inference_service = ExactFacialParalysisInference()
        print("✅ Exact inference service initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize exact inference service: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': inference_service is not None,
        'model_type': inference_service.model_info.get('model_type', 'Unknown') if inference_service else 'Unknown',
        'accuracy': inference_service.model_info.get('accuracy', 0.0) if inference_service else 0.0
    })

@app.route('/predict', methods=['POST'])
def predict_image():
    """Predict facial paralysis from uploaded image"""
    try:
        print(f"🚀 Received prediction request")
        
        if inference_service is None:
            print(f"❌ Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image data
        data = request.get_json()
        if 'image' not in data:
            print(f"❌ No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        print(f"✅ Image data received, length: {len(data['image'])}")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            print(f"✅ Image decoded successfully: {image.size}, {image.mode}")
        except Exception as e:
            print(f"❌ Image decode error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Make prediction
        print(f"🔍 Starting prediction...")
        result = inference_service.predict(image)
        if result is None:
            print(f"❌ Prediction returned None")
            return jsonify({'error': 'Prediction failed - check server logs'}), 500
        
        print(f"✅ Prediction successful: {result}")
        
        # Add model info
        result['model_info'] = {
            'model_type': inference_service.model_info.get('model_type', 'Unknown'),
            'accuracy': inference_service.model_info.get('accuracy', 0.0),
            'training_date': inference_service.model_info.get('training_date', 'Unknown')
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction endpoint error: {str(e)}")
        import traceback
        print(f"📋 Full traceback:")
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    if inference_service is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(inference_service.model_info)

if __name__ == '__main__':
    print("🚀 Starting EXACT Facial Paralysis Detection API...")
    
    if initialize_service():
        print("✅ Service ready!")
        print("📡 API Endpoints:")
        print("  GET  /health - Health check")
        print("  POST /predict - Predict from base64 image")
        print("  GET  /model_info - Get model information")
        print("\n🌐 Server starting on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to start service")
