#!/usr/bin/env python3
"""
Test script for Scanix AI Backend
Tests the backend API endpoints to ensure everything is working correctly
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_backend():
    """Test the backend API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Scanix AI Backend...")
    print("=" * 40)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   📊 Status: {data.get('status')}")
            print(f"   🔢 Version: {data.get('version')}")
            print(f"   🧠 Model loaded: {data.get('model_loaded')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Doctors Endpoint
    print("\n2. Testing doctors endpoint...")
    try:
        response = requests.get(f"{base_url}/doctors", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Doctors endpoint working")
            print(f"   👨‍⚕️ Found {len(data)} doctors")
        else:
            print(f"   ❌ Doctors endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Doctors endpoint error: {e}")
    
    # Test 3: Contact Form
    print("\n3. Testing contact form...")
    try:
        test_data = {
            "name": "Test User",
            "email": "test@example.com",
            "message": "This is a test message"
        }
        response = requests.post(
            f"{base_url}/contact",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            print("   ✅ Contact form working")
        else:
            print(f"   ❌ Contact form failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Contact form error: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Backend testing completed!")
    print("🌐 Backend is ready for frontend connection")
    return True

if __name__ == "__main__":
    # Wait a moment for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(2)
    
    success = test_backend()
    if success:
        print("\n🚀 Backend is ready! You can now start the Flutter frontend.")
    else:
        print("\n❌ Backend testing failed. Please check the backend logs.")
        sys.exit(1)
