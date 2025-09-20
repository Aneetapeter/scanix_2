#!/usr/bin/env python3
"""
Facial Paralysis Detection AI - Prediction Script
Use this script to detect facial paralysis in new images
"""

import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import json

class FacialParalysisDetector:
    def __init__(self, model_path='models/ai_model.pkl'):
        """Initialize the AI detector"""
        self.model = joblib.load(model_path)
        self.class_names = ['Normal', 'Paralysis']
        
        # Load model info
        with open('models/model_info.json', 'r') as f:
            self.model_info = json.load(f)
    
    def preprocess_image(self, image_path):
        """Preprocess an image for AI prediction"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to 32x32 (same as training)
                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and flatten
                img_array = np.array(img).flatten()
                
                return img_array.reshape(1, -1)
        except Exception as e:
            raise Exception(f"Error processing image: {e}")
    
    def predict(self, image_path):
        """Make a prediction on an image"""
        # Preprocess image
        X = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            'prediction': self.class_names[prediction],
            'confidence': float(max(probabilities)),
            'is_paralysis': bool(prediction),
            'probabilities': {
                'normal': float(probabilities[0]),
                'paralysis': float(probabilities[1])
            }
        }
    
    def predict_batch(self, image_paths):
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        return results
    
    def get_model_info(self):
        """Get information about the trained model"""
        return self.model_info

def main():
    """Example usage of the AI detector"""
    print("🤖 Facial Paralysis Detection AI")
    print("=" * 40)
    
    # Initialize detector
    try:
        detector = FacialParalysisDetector()
        print("✅ AI model loaded successfully!")
        
        # Show model info
        info = detector.get_model_info()
        print(f"Model Accuracy: {info['accuracy']:.4f}")
        print(f"Total Training Images: {info['total_images']}")
        print(f"Normal Images: {info['normal_images']}")
        print(f"Paralysis Images: {info['paralysis_images']}")
        
        # Example prediction (you can replace this with your image path)
        example_image = "data/test/normal/normal_0000_Aaron_Eckhart_Aaron_Eckhart_0001.jpg"
        
        if Path(example_image).exists():
            print(f"\n🔍 Testing with example image: {example_image}")
            result = detector.predict(example_image)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Is Paralysis: {result['is_paralysis']}")
            print(f"Probabilities:")
            print(f"  Normal: {result['probabilities']['normal']:.4f}")
            print(f"  Paralysis: {result['probabilities']['paralysis']:.4f}")
        else:
            print("\n📝 To test with your own image:")
            print("1. Place your image in the backend directory")
            print("2. Run: python predict_facial_paralysis.py")
            print("3. Or use the detector in your code:")
            print("   detector = FacialParalysisDetector()")
            print("   result = detector.predict('your_image.jpg')")
        
    except Exception as e:
        print(f"❌ Error loading AI model: {e}")
        print("Make sure you've trained the model first by running: python quick_train.py")

if __name__ == "__main__":
    main()
