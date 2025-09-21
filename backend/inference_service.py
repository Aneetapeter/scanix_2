#!/usr/bin/env python3
"""
Working Inference Service for Facial Paralysis Detection
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

class FacialParalysisInference:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.ml_model = None
        self.scaler = None
        self.model_info = None
        self.image_size = (64, 64)
        
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
    
    def extract_features(self, image):
        """Extract features EXACTLY as in training"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to match training size (64, 64)
            image = image.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            features = []
            
            # 1. Basic pixel features (all pixels)
            basic_features = gray_array.flatten()
            features.extend(basic_features)
            
            # 2. Asymmetry features (MOST IMPORTANT for paralysis)
            left_half = gray_array[:, :32]
            right_half = gray_array[:, 32:]
            right_flipped = np.fliplr(right_half)
            
            # Multiple asymmetry metrics
            asymmetry_mean = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_std = np.std(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_max = np.max(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            features.extend([asymmetry_mean, asymmetry_std, asymmetry_max])
            
            # 3. Eye region asymmetry (critical for paralysis)
            eye_region_left = gray_array[20:40, 15:45]  # Left eye region
            eye_region_right = gray_array[20:40, 19:49]  # Right eye region
            eye_region_right_flipped = np.fliplr(eye_region_right)
            
            eye_asymmetry = np.mean(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
            features.append(eye_asymmetry)
            
            # 4. Mouth region asymmetry
            mouth_region_left = gray_array[45:60, 20:40]  # Left mouth region
            mouth_region_right = gray_array[45:60, 24:44]  # Right mouth region
            mouth_region_right_flipped = np.fliplr(mouth_region_right)
            
            mouth_asymmetry = np.mean(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
            features.append(mouth_asymmetry)
            
            # 5. Edge features
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (64 * 64)
            features.append(edge_density)
            
            # 6. Statistical features
            stats_features = [
                np.mean(gray_array),
                np.std(gray_array),
                np.var(gray_array),
                np.median(gray_array),
                np.percentile(gray_array, 25),
                np.percentile(gray_array, 75)
            ]
            features.extend(stats_features)
            
            # Convert to numpy array
            features_array = np.array(features)
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error in feature extraction: {e}")
            return None
    
    def predict(self, image):
        """Predict facial paralysis from image"""
        try:
            # Extract features
            features = self.extract_features(image)
            if features is None:
                return None
            
            # Check if feature count matches model expectations
            expected_features = self.ml_model.n_features_in_
            if len(features) != expected_features:
                print(f"❌ Feature count mismatch! Expected: {expected_features}, Got: {len(features)}")
                return None
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.ml_model.predict(features_scaled)[0]
            probability = self.ml_model.predict_proba(features_scaled)[0]
            
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
        inference_service = FacialParalysisInference()
        print("✅ Inference service initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize inference service: {e}")
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
        if inference_service is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image data
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Make prediction
        result = inference_service.predict(image)
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Add model info
        result['model_info'] = {
            'model_type': inference_service.model_info.get('model_type', 'Unknown'),
            'accuracy': inference_service.model_info.get('accuracy', 0.0),
            'training_date': inference_service.model_info.get('training_date', 'Unknown')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    if inference_service is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(inference_service.model_info)

if __name__ == '__main__':
    print("🚀 Starting Facial Paralysis Detection API...")
    
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
