#!/usr/bin/env python3
"""
Exact Inference Service for Facial Paralysis Detection
Uses the EXACT same feature extraction as training
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
        """Extract features EXACTLY as in training"""
        try:
            print(f"🔍 Starting EXACT feature extraction...")
            print(f"   Original image size: {image.size}")
            print(f"   Original image mode: {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to EXACT training size
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            print(f"   Resized image shape: {gray_array.shape}")
            
            features = []
            
            # 1. Facial Asymmetry Features (Most Important) - EXACT from training
            left_half = gray_array[:, :64]
            right_half = gray_array[:, 64:]
            right_flipped = np.fliplr(right_half)
            
            # Multiple asymmetry metrics
            asymmetry_mean = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_std = np.std(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_max = np.max(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_median = np.median(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            
            features.extend([asymmetry_mean, asymmetry_std, asymmetry_max, asymmetry_median])
            
            # 2. Eye Region Asymmetry (Critical for paralysis) - EXACT from training
            eye_region_left = gray_array[30:60, 20:60]  # Left eye region
            eye_region_right = gray_array[30:60, 68:108]  # Right eye region
            eye_region_right_flipped = np.fliplr(eye_region_right)
            
            eye_asymmetry_mean = np.mean(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
            eye_asymmetry_std = np.std(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
            features.extend([eye_asymmetry_mean, eye_asymmetry_std])
            
            # 3. Mouth Region Asymmetry - EXACT from training
            mouth_region_left = gray_array[80:110, 40:80]  # Left mouth region
            mouth_region_right = gray_array[80:110, 48:88]  # Right mouth region
            mouth_region_right_flipped = np.fliplr(mouth_region_right)
            
            mouth_asymmetry_mean = np.mean(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
            mouth_asymmetry_std = np.std(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
            features.extend([mouth_asymmetry_mean, mouth_asymmetry_std])
            
            # 4. Edge Asymmetry (using simple gradient) - EXACT from training
            left_edges_x = np.abs(np.diff(left_half, axis=1))
            left_edges_y = np.abs(np.diff(left_half, axis=0))
            right_edges_x = np.abs(np.diff(right_half, axis=1))
            right_edges_y = np.abs(np.diff(right_half, axis=0))
            right_edges_x_flipped = np.fliplr(right_edges_x)
            right_edges_y_flipped = np.fliplr(right_edges_y)
            
            edge_asymmetry_x = np.mean(np.abs(left_edges_x.astype(float) - right_edges_x_flipped.astype(float)))
            edge_asymmetry_y = np.mean(np.abs(left_edges_y.astype(float) - right_edges_y_flipped.astype(float)))
            features.extend([edge_asymmetry_x, edge_asymmetry_y])
            
            # 5. Texture Asymmetry (using Laplacian approximation) - EXACT from training
            left_texture = np.abs(np.diff(left_half, axis=1)) + np.abs(np.diff(left_half, axis=0))
            right_texture = np.abs(np.diff(right_half, axis=1)) + np.abs(np.diff(right_half, axis=0))
            right_texture_flipped = np.fliplr(right_texture)
            
            texture_asymmetry = np.mean(np.abs(left_texture - right_texture_flipped))
            features.append(texture_asymmetry)
            
            # 6. Histogram Asymmetry - EXACT from training
            left_hist, _ = np.histogram(left_half.flatten(), bins=32, range=(0, 256))
            right_hist, _ = np.histogram(right_half.flatten(), bins=32, range=(0, 256))
            
            hist_correlation = np.corrcoef(left_hist, right_hist)[0, 1]
            hist_chi_square = np.sum((left_hist - right_hist) ** 2 / (right_hist + 1e-8))
            features.extend([hist_correlation, hist_chi_square])
            
            # 7. Regional Asymmetry (divide face into regions) - EXACT from training
            regions = [
                (0, 32, 0, 64),    # Top left
                (0, 32, 64, 128),  # Top right
                (32, 64, 0, 64),   # Mid left
                (32, 64, 64, 128), # Mid right
                (64, 96, 0, 64),   # Lower left
                (64, 96, 64, 128), # Lower right
            ]
            
            for y1, y2, x1, x2 in regions:
                region_left = gray_array[y1:y2, x1:x2]
                region_right = gray_array[y1:y2, x1+64:x2+64] if x2 <= 64 else gray_array[y1:y2, x1-64:x2-64]
                if region_right.shape == region_left.shape:
                    region_right_flipped = np.fliplr(region_right)
                    region_asymmetry = np.mean(np.abs(region_left.astype(float) - region_right_flipped.astype(float)))
                    features.append(region_asymmetry)
            
            # 8. Basic pixel features (sampled) - EXACT from training
            # Sample every 4th pixel to reduce dimensionality
            sampled_pixels = gray_array[::4, ::4].flatten()
            features.extend(sampled_pixels)
            
            # 9. Statistical features - EXACT from training
            stats_features = [
                np.mean(gray_array),
                np.std(gray_array),
                np.var(gray_array),
                np.median(gray_array),
                np.percentile(gray_array, 25),
                np.percentile(gray_array, 75),
                np.percentile(gray_array, 90),
                np.percentile(gray_array, 95),
                np.skew(gray_array.flatten()),
                np.kurtosis(gray_array.flatten())
            ]
            features.extend(stats_features)
            
            # Convert to numpy array
            features_array = np.array(features)
            
            print(f"✅ EXACT feature extraction completed")
            print(f"   Total features: {len(features_array)}")
            print(f"   Expected features: {self.ml_model.n_features_in_}")
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error in EXACT feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, image):
        """Predict facial paralysis from image"""
        try:
            print(f"🚀 Starting prediction...")
            
            # Extract features using EXACT training method
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
