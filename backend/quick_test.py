#!/usr/bin/env python3
"""
Quick test to verify API endpoints
"""

import requests
import base64
from pathlib import Path
from PIL import Image
import io

def test_endpoints():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing API Endpoints...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test with a simple image
    try:
        # Create a simple test image
        img = Image.new('RGB', (128, 128), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # Test /predict endpoint
        print("\n🔍 Testing /predict endpoint...")
        response = requests.post(
            f"{base_url}/predict",
            json={"image": img_base64},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ /predict endpoint working")
            print(f"   Prediction: {data.get('prediction', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}")
        else:
            print(f"❌ /predict failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing /predict: {e}")
    
    print("\n✅ API endpoint test completed!")

if __name__ == "__main__":
    test_endpoints()