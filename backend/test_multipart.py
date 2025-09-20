#!/usr/bin/env python3
"""
Test script for multipart image upload to the fixed AI server
"""

import requests
from pathlib import Path

def test_multipart_upload():
    """Test multipart file upload to the analyze endpoint"""
    print("🧪 Testing Multipart Image Upload...")
    print("=" * 50)
    
    # Find a test image
    test_image_path = None
    for class_name in ['normal', 'paralysis']:
        test_dir = Path(f'data/test/{class_name}')
        if test_dir.exists():
            images = list(test_dir.glob('*.jpg'))
            if images:
                test_image_path = images[0]
                break
    
    if not test_image_path:
        print("❌ No test images found!")
        return
    
    print(f"📸 Using test image: {test_image_path}")
    
    # Test multipart upload
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://127.0.0.1:5000/analyze', files=files)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Is Paralysis: {result['is_paralysis']}")
            print(f"   Probabilities:")
            print(f"     Normal: {result['probabilities']['normal']:.4f}")
            print(f"     Paralysis: {result['probabilities']['paralysis']:.4f}")
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_multipart_upload()
