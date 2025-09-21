#!/usr/bin/env python3
"""
Test the final model on external images
"""

import joblib
import numpy as np
from pathlib import Path
from PIL import Image
import json

def extract_features(image):
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
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def test_external_image(image_path, model, scaler):
    """Test a single external image"""
    try:
        print(f"\n🔍 Testing image: {image_path}")
        
        with Image.open(image_path) as img:
            print(f"   Original size: {img.size}")
            print(f"   Original mode: {img.mode}")
            
            # Extract features
            features = extract_features(img)
            if features is None:
                print("❌ Feature extraction failed")
                return None
            
            print(f"   Features extracted: {len(features)}")
            
            # Check feature count
            expected_features = model.n_features_in_
            if len(features) != expected_features:
                print(f"❌ Feature count mismatch! Expected: {expected_features}, Got: {len(features)}")
                return None
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            result = {
                'image_path': str(image_path),
                'prediction': int(prediction),
                'confidence': float(max(probability)),
                'probabilities': {
                    'normal': float(probability[0]),
                    'paralysis': float(probability[1])
                },
                'label': 'Normal Face' if prediction == 0 else 'Paralyzed Face (Droopy)',
                'description': 'The face appears to be normal with no signs of paralysis.' if prediction == 0 else 'The face shows signs of facial paralysis or drooping.'
            }
            
            print(f"✅ Prediction: {result['label']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Normal probability: {result['probabilities']['normal']:.3f}")
            print(f"   Paralysis probability: {result['probabilities']['paralysis']:.3f}")
            
            return result
            
    except Exception as e:
        print(f"❌ Error testing {image_path}: {e}")
        return None

def main():
    """Test external images"""
    print("🧪 Testing External Images with Final Model")
    print("=" * 50)
    
    # Load model
    try:
        model = joblib.load('models/ai_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        print("✅ Model loaded successfully")
        print(f"   Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"   Accuracy: {model_info.get('accuracy', 0.0):.4f}")
        print(f"   ROC AUC: {model_info.get('roc_auc', 0.0):.4f}")
        print(f"   Expected features: {model.n_features_in_}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test external images
    external_dirs = [
        'data/test/normal',
        'data/test/paralysis',
        'data/validation/normal',
        'data/validation/paralysis'
    ]
    
    results = []
    
    for test_dir in external_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            print(f"\n📁 Testing images from: {test_dir}")
            
            for img_path in test_path.glob('*.jpg'):
                result = test_external_image(img_path, model, scaler)
                if result:
                    results.append(result)
                
                # Limit to first 3 images per directory for testing
                if len([r for r in results if test_dir in r['image_path']]) >= 3:
                    break
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 External Image Test Summary:")
    
    if results:
        normal_predictions = [r for r in results if r['prediction'] == 0]
        paralysis_predictions = [r for r in results if r['prediction'] == 1]
        
        print(f"   Total images tested: {len(results)}")
        print(f"   Predicted as normal: {len(normal_predictions)}")
        print(f"   Predicted as paralysis: {len(paralysis_predictions)}")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        print(f"\n🔍 Detailed Results:")
        for result in results:
            print(f"   {Path(result['image_path']).name}: {result['label']} (confidence: {result['confidence']:.3f})")
    else:
        print("❌ No external images found or tested")

if __name__ == "__main__":
    main()
