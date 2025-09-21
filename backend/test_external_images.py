#!/usr/bin/env python3
"""
Test script to validate model performance on external images
"""

import joblib
import numpy as np
from pathlib import Path
from PIL import Image
import json
import base64
import io

def extract_robust_features(image):
    """Extract features matching the robust training"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 128x128 (matching robust training)
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = image.convert('L')
        gray_array = np.array(gray)
        
        features = []
        
        # 1. Facial Asymmetry Features (Most Important for paralysis)
        left_half = gray_array[:, :64]
        right_half = gray_array[:, 64:]
        right_flipped = np.fliplr(right_half)
        
        # Multiple asymmetry metrics
        asymmetry_mean = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        asymmetry_std = np.std(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        asymmetry_max = np.max(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        asymmetry_median = np.median(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        
        features.extend([asymmetry_mean, asymmetry_std, asymmetry_max, asymmetry_median])
        
        # 2. Eye Region Asymmetry (Critical for paralysis detection)
        eye_region_left = gray_array[30:60, 20:60]  # Left eye region
        eye_region_right = gray_array[30:60, 68:108]  # Right eye region
        eye_region_right_flipped = np.fliplr(eye_region_right)
        
        eye_asymmetry_mean = np.mean(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
        eye_asymmetry_std = np.std(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
        features.extend([eye_asymmetry_mean, eye_asymmetry_std])
        
        # 3. Mouth Region Asymmetry
        mouth_region_left = gray_array[80:110, 40:80]  # Left mouth region
        mouth_region_right = gray_array[80:110, 48:88]  # Right mouth region
        mouth_region_right_flipped = np.fliplr(mouth_region_right)
        
        mouth_asymmetry_mean = np.mean(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
        mouth_asymmetry_std = np.std(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
        features.extend([mouth_asymmetry_mean, mouth_asymmetry_std])
        
        # 4. Edge Asymmetry
        left_edges_x = np.abs(np.diff(left_half, axis=1))
        left_edges_y = np.abs(np.diff(left_half, axis=0))
        right_edges_x = np.abs(np.diff(right_half, axis=1))
        right_edges_y = np.abs(np.diff(right_half, axis=0))
        right_edges_x_flipped = np.fliplr(right_edges_x)
        right_edges_y_flipped = np.fliplr(right_edges_y)
        
        edge_asymmetry_x = np.mean(np.abs(left_edges_x.astype(float) - right_edges_x_flipped.astype(float)))
        edge_asymmetry_y = np.mean(np.abs(left_edges_y.astype(float) - right_edges_y_flipped.astype(float)))
        features.extend([edge_asymmetry_x, edge_asymmetry_y])
        
        # 5. Texture Asymmetry
        left_texture = np.abs(np.diff(left_half, axis=1)) + np.abs(np.diff(left_half, axis=0))
        right_texture = np.abs(np.diff(right_half, axis=1)) + np.abs(np.diff(right_half, axis=0))
        right_texture_flipped = np.fliplr(right_texture)
        
        texture_asymmetry = np.mean(np.abs(left_texture - right_texture_flipped))
        features.append(texture_asymmetry)
        
        # 6. Histogram Asymmetry
        left_hist, _ = np.histogram(left_half.flatten(), bins=32, range=(0, 256))
        right_hist, _ = np.histogram(right_half.flatten(), bins=32, range=(0, 256))
        
        hist_correlation = np.corrcoef(left_hist, right_hist)[0, 1]
        hist_chi_square = np.sum((left_hist - right_hist) ** 2 / (right_hist + 1e-8))
        features.extend([hist_correlation, hist_chi_square])
        
        # 7. Regional Asymmetry (divide face into regions)
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
        
        # 8. Statistical features
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
        
        # 9. Global features (reduced pixel sampling for robustness)
        # Sample every 8th pixel to reduce overfitting
        sampled_pixels = gray_array[::8, ::8].flatten()
        features.extend(sampled_pixels)
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def test_external_image(image_path, model, scaler):
    """Test a single external image"""
    try:
        print(f"\n🔍 Testing external image: {image_path}")
        
        with Image.open(image_path) as img:
            print(f"   Original size: {img.size}")
            print(f"   Original mode: {img.mode}")
            
            # Extract features
            features = extract_robust_features(img)
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
    print("🧪 Testing External Images with Current Model")
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
                
                # Limit to first 5 images per directory for testing
                if len([r for r in results if test_dir in r['image_path']]) >= 5:
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
