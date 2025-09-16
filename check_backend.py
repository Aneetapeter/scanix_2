#!/usr/bin/env python3
"""
Check if Scanix backend is running and accessible
"""

import requests
import sys
import time

def check_backend():
    """Check if backend is running and accessible"""
    urls = [
        'http://localhost:5000/health',
        'http://127.0.0.1:5000/health',
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend is running at {url}")
                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   Model loaded: {data.get('model_loaded', False)}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to {url}: {e}")
    
    return False

def main():
    print("Scanix Backend Health Check")
    print("=" * 30)
    
    if check_backend():
        print("\n✅ Backend is ready!")
        print("You can now run the Flutter app:")
        print("  flutter run -d web-server --web-port 3000")
    else:
        print("\n❌ Backend is not running!")
        print("\nTo start the backend:")
        print("  1. cd backend")
        print("  2. pip install -r requirements.txt")
        print("  3. python run.py")
        print("\nOr run: start_dev.bat (Windows) or start_dev.sh (macOS/Linux)")
        sys.exit(1)

if __name__ == '__main__':
    main()
