#!/usr/bin/env python3
"""
Quick test to verify the model works
"""

import joblib
import numpy as np
from pathlib import Path
from PIL import Image

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

def main():
    """Quick test"""
    print("🧪 Quick Test of Final Model")
    print("=" * 40)
    
    # Load model
    try:
        model = joblib.load('models/ai_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        print("✅ Model loaded successfully")
        print(f"   Expected features: {model.n_features_in_}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test on a few images
    test_dirs = ['data/test/normal', 'data/test/paralysis']
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            print(f"\n📁 Testing images from: {test_dir}")
            
            count = 0
            for img_path in test_path.glob('*.jpg'):
                if count >= 2:  # Test only 2 images per directory
                    break
                
                try:
                    with Image.open(img_path) as img:
                        features = extract_features(img)
                        if features is not None:
                            features_scaled = scaler.transform([features])
                            prediction = model.predict(features_scaled)[0]
                            probability = model.predict_proba(features_scaled)[0]
                            
                            label = 'Normal' if prediction == 0 else 'Paralysis'
                            confidence = max(probability)
                            
                            print(f"   {img_path.name}: {label} (confidence: {confidence:.3f})")
                            count += 1
                            
                except Exception as e:
                    print(f"   Error with {img_path.name}: {e}")
                    continue
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    main()
