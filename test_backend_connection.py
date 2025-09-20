#!/usr/bin/env python3
"""
Test script to verify backend connection and API endpoints
"""

import requests
import json
import time

def test_backend_connection():
    """Test all backend endpoints"""
    base_url = "http://localhost:5000"
    
    print("🔍 Testing Backend Connection...")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Timestamp: {data['timestamp']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False
    
    # Test 2: Doctors Endpoint
    print("\n2. Testing Doctors Endpoint...")
    try:
        response = requests.get(f"{base_url}/doctors", timeout=5)
        if response.status_code == 200:
            doctors = response.json()
            print(f"✅ Doctors Endpoint: {len(doctors)} doctors found")
            for i, doctor in enumerate(doctors[:2], 1):  # Show first 2 doctors
                print(f"   {i}. {doctor['name']} - {doctor['specialization']}")
        else:
            print(f"❌ Doctors Endpoint Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Doctors Endpoint Error: {e}")
        return False
    
    # Test 3: Contact Form
    print("\n3. Testing Contact Form...")
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
            data = response.json()
            print(f"✅ Contact Form: {data['message']}")
        else:
            print(f"❌ Contact Form Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Contact Form Error: {e}")
        return False
    
    # Test 4: Send Report
    print("\n4. Testing Send Report...")
    try:
        test_data = {
            "doctor_id": "1",
            "result": {
                "has_paralysis": False,
                "confidence": 0.85,
                "recommendation": "Test recommendation",
                "timestamp": "2024-01-01T00:00:00"
            }
        }
        response = requests.post(
            f"{base_url}/send-report",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Send Report: {data['message']}")
        else:
            print(f"❌ Send Report Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Send Report Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All Backend Tests Passed!")
    print("✅ Backend is ready for frontend connection")
    return True

if __name__ == "__main__":
    print("🚀 Starting Backend Connection Test...")
    success = test_backend_connection()
    
    if success:
        print("\n📱 You can now run your Flutter app:")
        print("   flutter run -d web-server --web-port 3000")
        print("\n🌐 Access the app at: http://localhost:3000")
    else:
        print("\n❌ Backend connection failed. Please check the server.")
