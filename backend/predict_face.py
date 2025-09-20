#!/usr/bin/env python3
"""
Face Classification Prediction Script
Uses the trained model to predict if a face is normal or paralyzed
"""

import sys
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
from train_new_face_model import FaceClassificationTrainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacePredictor:
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
            import json
            with open(self.model_dir / 'model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Create feature extractor
            self.feature_extractor = FaceClassificationTrainer()
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Model type: {self.model_info.get('model_type', 'Unknown')}")
            logger.info(f"   Feature count: {self.model.n_features_in_}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def predict_image(self, image_path):
        """Predict if an image shows a normal or paralyzed face"""
        try:
            logger.info(f"🔍 Analyzing image: {image_path}")
            
            # Load image
            with Image.open(image_path) as img:
                logger.info(f"   Original size: {img.size}")
                logger.info(f"   Original mode: {img.mode}")
                
                # Extract features
                features = self.feature_extractor.extract_face_features(img)
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
                    'prediction': int(prediction),
                    'label': class_labels.get(prediction, 'unknown'),
                    'confidence': float(max(probabilities)),
                    'probabilities': {
                        'normal': float(probabilities[0]),
                        'paralyzed': float(probabilities[1])
                    }
                }
                
                logger.info(f"✅ Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python predict_face.py <image_path>")
        print("Example: python predict_face.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        # Create predictor
        predictor = FacePredictor()
        
        # Make prediction
        result = predictor.predict_image(image_path)
        
        if result:
            print("\n" + "=" * 50)
            print("🎯 FACE CLASSIFICATION RESULT")
            print("=" * 50)
            print(f"📸 Image: {image_path}")
            print(f"🔍 Prediction: {result['label'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"📈 Probabilities:")
            print(f"   Normal: {result['probabilities']['normal']:.1%}")
            print(f"   Paralyzed: {result['probabilities']['paralyzed']:.1%}")
            print("=" * 50)
        else:
            print("❌ Prediction failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
