#!/usr/bin/env python3
"""
Test current model performance
"""

import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def extract_enhanced_features(image):
    """Extract enhanced features matching the improved training"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 64x64 (matching improved training)
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = image.convert('L')
        gray_array = np.array(gray)
        
        # Extract multiple feature types
        features = []
        
        # 1. Basic pixel features
        basic_features = gray_array.flatten()
        features.extend(basic_features)
        
        # 2. Asymmetry features (crucial for paralysis detection)
        left_half = gray_array[:, :32]
        right_half = gray_array[:, 32:]
        right_flipped = np.fliplr(right_half)
        
        # Calculate asymmetry metrics
        asymmetry = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        features.append(asymmetry)
        
        # 3. Edge features
        grad_x = np.abs(np.diff(gray_array, axis=1))
        grad_y = np.abs(np.diff(gray_array, axis=0))
        edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (64 * 64)
        features.append(edge_density)
        
        # 4. Texture features
        texture_features = []
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                block = gray_array[i:i+8, j:j+8]
                if block.size > 0:
                    features.extend([np.mean(block), np.std(block)])
                    if block.shape[0] > 1 and block.shape[1] > 1:
                        grad_x = np.abs(np.diff(block, axis=1))
                        grad_y = np.abs(np.diff(block, axis=0))
                        features.extend([np.mean(grad_x), np.mean(grad_y)])
        
        # 5. Statistical features
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
        raise Exception(f"Feature extraction error: {e}")

def test_current_model():
    """Test the current model on validation data"""
    print("🧪 Testing improved model performance...")
    
    # Load model and scaler
    try:
        model = joblib.load('models/ai_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model and scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load validation data
    data_dir = Path('data/validation')
    X_test = []
    y_test = []
    
    # Load normal images with improved feature extraction
    normal_dir = data_dir / 'normal'
    if normal_dir.exists():
        for img_path in normal_dir.glob('*.jpg'):
            try:
                with Image.open(img_path) as img:
                    # Use improved feature extraction
                    features = extract_enhanced_features(img)
                    X_test.append(features)
                    y_test.append(0)  # Normal
            except:
                continue
    
    # Load paralysis images with improved feature extraction
    paralysis_dir = data_dir / 'paralysis'
    if paralysis_dir.exists():
        for img_path in paralysis_dir.glob('*.jpg'):
            try:
                with Image.open(img_path) as img:
                    # Use improved feature extraction
                    features = extract_enhanced_features(img)
                    X_test.append(features)
                    y_test.append(1)  # Paralysis
            except:
                continue
    
    if len(X_test) == 0:
        print("❌ No validation data found")
        return
    
    print(f"✅ Loaded {len(X_test)} validation images")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"📊 Testing on {len(X_test)} validation images")
    print(f"   Normal: {sum(y_test == 0)}")
    print(f"   Paralysis: {sum(y_test == 1)}")
    
    # Scale features and make predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"📈 Current model accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Paralysis']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Normal  Paralysis")
    print(f"Actual Normal    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Paralysis {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Analyze errors
    normal_errors = np.sum((y_test == 0) & (y_pred == 1))
    paralysis_errors = np.sum((y_test == 1) & (y_pred == 0))
    
    print(f"\n❌ Error Analysis:")
    print(f"   Normal images misclassified as paralysis: {normal_errors}")
    print(f"   Paralysis images misclassified as normal: {paralysis_errors}")
    
    if normal_errors > paralysis_errors:
        print("   ⚠️  Model is biased toward predicting paralysis")
    elif paralysis_errors > normal_errors:
        print("   ⚠️  Model is biased toward predicting normal")
    else:
        print("   ✅ Model shows balanced performance")

if __name__ == "__main__":
    test_current_model()
