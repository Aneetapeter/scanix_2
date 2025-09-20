#!/usr/bin/env python3
"""
Face Classification API
Flask API for face classification using the trained model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import json
import joblib
from pathlib import Path
import logging
from train_new_face_model import FaceClassificationTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:3000', 'http://127.0.0.1:3000',
    'http://localhost:3001', 'http://127.0.0.1:3001',
    'http://localhost:3002', 'http://127.0.0.1:3002',
    'http://localhost:8080', 'http://127.0.0.1:8080'
])

class FaceClassificationAPI:
    def __init__(self, model_dir="models_new"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.model_info = None
        self.feature_extractor = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load model
            self.model = joblib.load(self.model_dir / 'face_classification_model.pkl')
            
            # Load scaler
            self.scaler = joblib.load(self.model_dir / 'face_scaler.pkl')
            
            # Load model info
            with open(self.model_dir / 'model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Create feature extractor
            self.feature_extractor = FaceClassificationTrainer()
            
            logger.info("✅ Face classification model loaded successfully")
            logger.info(f"   Model type: {self.model_info.get('model_type', 'Unknown')}")
            logger.info(f"   Feature count: {self.model.n_features_in_}")
            
        except Exception as e:
            logger.error(f"❌ Error loading face classification model: {e}")
            raise
    
    def predict_image(self, image_data):
        """Predict if an image shows a normal or paralyzed face"""
        try:
            logger.info("🔍 Starting face classification...")
            
            # Handle different input formats
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.info(f"   Image size: {image.size}")
            logger.info(f"   Image mode: {image.mode}")
            
            # Extract features
            features = self.feature_extractor.extract_face_features(image)
            if features is None:
                raise ValueError("Feature extraction failed")
            
            logger.info(f"   Features extracted: {len(features)}")
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get class labels
            class_labels = self.model_info.get('class_labels', {0: 'normal', 1: 'paralyzed'})
            
            # Format results
            result = {
                'success': True,
                'prediction': int(prediction),
                'label': class_labels.get(prediction, 'unknown'),
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'paralyzed': float(probabilities[1])
                },
                'description': self.get_description(prediction, max(probabilities))
            }
            
            logger.info(f"✅ Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Face classification error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_description(self, prediction, confidence):
        """Get a description based on the prediction"""
        if prediction == 1:  # Paralyzed
            if confidence > 0.8:
                return "High confidence of facial paralysis detected. Medical consultation recommended."
            else:
                return "Possible facial paralysis detected. Consider medical evaluation."
        else:  # Normal
            return "No signs of facial paralysis detected. Face appears normal."

# Initialize API
api = None

def initialize_api():
    """Initialize the face classification API"""
    global api
    try:
        api = FaceClassificationAPI()
        logger.info("✅ Face classification API initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize face classification API: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': api is not None,
        'model_type': api.model_info.get('model_type', 'Unknown') if api else 'Unknown'
    })

@app.route('/classify', methods=['POST'])
def classify_face():
    """Classify face as normal or paralyzed"""
    try:
        logger.info("🚀 Received face classification request")
        
        if api is None:
            logger.error("❌ Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image data
        data = request.get_json()
        if 'image' not in data:
            logger.error("❌ No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        logger.info(f"✅ Image data received, length: {len(data['image'])}")
        
        # Make prediction
        result = api.predict_image(data['image'])
        
        if not result['success']:
            logger.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            return jsonify({'error': result.get('error', 'Prediction failed')}), 500
        
        logger.info(f"✅ Classification successful: {result['label']}")
        
        # Add model info
        result['model_info'] = {
            'model_type': api.model_info.get('model_type', 'Unknown'),
            'training_date': api.model_info.get('training_date', 'Unknown'),
            'feature_count': api.model_info.get('feature_count', 'Unknown')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Classification endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    if api is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(api.model_info)

if __name__ == '__main__':
    print("🚀 Starting Face Classification API...")
    
    if initialize_api():
        print("✅ API ready!")
        print("📡 API Endpoints:")
        print("  GET  /health - Health check")
        print("  POST /classify - Classify face as normal/paralyzed")
        print("  GET  /model_info - Get model information")
        print("\n🌐 Server starting on http://localhost:5001")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("❌ Failed to start API")
