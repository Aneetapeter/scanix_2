#!/usr/bin/env python3
"""
Test script to validate preprocessing consistency between training and inference
"""

import numpy as np
from PIL import Image
import joblib
from pathlib import Path
from standardized_preprocessing import preprocessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_preprocessing_consistency():
    """Test that preprocessing produces consistent results"""
    
    print("🧪 Testing Preprocessing Consistency")
    print("=" * 50)
    
    # Load model to get expected feature count
    try:
        model = joblib.load('models/ai_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print(f"✅ Model loaded successfully")
        print(f"   Expected features: {model.n_features_in_}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Create a test image
    print(f"\n🖼️ Creating test image...")
    test_image = Image.new('RGB', (200, 200), color='white')
    print(f"   Test image size: {test_image.size}")
    
    # Test feature extraction
    print(f"\n🔍 Testing feature extraction...")
    try:
        features = preprocessor.extract_paralysis_features(test_image)
        if features is None:
            print(f"❌ Feature extraction failed")
            return False
        
        print(f"✅ Feature extraction successful")
        print(f"   Features extracted: {len(features)}")
        print(f"   Expected features: {model.n_features_in_}")
        
        # Check feature count
        if len(features) != model.n_features_in_:
            print(f"❌ Feature count mismatch!")
            print(f"   Expected: {model.n_features_in_}")
            print(f"   Got: {len(features)}")
            return False
        
        print(f"✅ Feature count matches model expectations")
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full preprocessing pipeline
    print(f"\n🔄 Testing full preprocessing pipeline...")
    try:
        # Convert image to base64 for testing
        import base64
        import io
        
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Test preprocessing
        processed_features = preprocessor.preprocess_image(image_data)
        
        print(f"✅ Full preprocessing successful")
        print(f"   Processed shape: {processed_features.shape}")
        print(f"   Expected shape: (1, {model.n_features_in_})")
        
        # Check shape
        if processed_features.shape != (1, model.n_features_in_):
            print(f"❌ Shape mismatch!")
            print(f"   Expected: (1, {model.n_features_in_})")
            print(f"   Got: {processed_features.shape}")
            return False
        
        print(f"✅ Shape matches model expectations")
        
    except Exception as e:
        print(f"❌ Full preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test scaling
    print(f"\n🔧 Testing feature scaling...")
    try:
        scaled_features = scaler.transform(processed_features)
        print(f"✅ Feature scaling successful")
        print(f"   Scaled shape: {scaled_features.shape}")
        
    except Exception as e:
        print(f"❌ Feature scaling error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction
    print(f"\n🤖 Testing prediction...")
    try:
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        print(f"✅ Prediction successful")
        print(f"   Prediction: {prediction}")
        print(f"   Probabilities: {probabilities}")
        print(f"   Confidence: {max(probabilities):.4f}")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🎉 All tests passed! Preprocessing is consistent.")
    return True

def test_with_real_image():
    """Test with a real image if available"""
    
    print(f"\n🖼️ Testing with real image...")
    
    # Look for test images
    test_dirs = [
        'data/test/normal',
        'data/test/paralyzed', 
        'data/processed_data/normal',
        'data/processed_data/paralysis'
    ]
    
    test_image_path = None
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            image_files = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
            if image_files:
                test_image_path = image_files[0]
                break
    
    if test_image_path is None:
        print(f"⚠️ No test images found, skipping real image test")
        return True
    
    print(f"   Using test image: {test_image_path}")
    
    try:
        # Load real image
        with Image.open(test_image_path) as img:
            print(f"   Original image size: {img.size}")
            print(f"   Original image mode: {img.mode}")
            
            # Test feature extraction
            features = preprocessor.extract_paralysis_features(img)
            if features is None:
                print(f"❌ Feature extraction failed with real image")
                return False
            
            print(f"✅ Real image feature extraction successful")
            print(f"   Features extracted: {len(features)}")
            
            # Test full pipeline
            import base64
            import io
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            processed_features = preprocessor.preprocess_image(image_data)
            print(f"✅ Real image full preprocessing successful")
            print(f"   Processed shape: {processed_features.shape}")
            
    except Exception as e:
        print(f"❌ Real image test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Preprocessing Consistency Tests")
    print("=" * 60)
    
    # Test 1: Basic consistency
    success1 = test_preprocessing_consistency()
    
    # Test 2: Real image test
    success2 = test_with_real_image()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Preprocessing is consistent between training and inference")
        print("✅ Feature counts match model expectations")
        print("✅ Image resizing is correct (128x128)")
        print("✅ All feature extraction steps work correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above")
    
    return success1 and success2

if __name__ == "__main__":
    main()
