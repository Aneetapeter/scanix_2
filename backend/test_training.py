#!/usr/bin/env python3

print("🚀 Starting test...")

try:
    import os
    print("✅ OS imported")
    
    import numpy as np
    print("✅ NumPy imported")
    
    from PIL import Image
    print("✅ PIL imported")
    
    import joblib
    print("✅ Joblib imported")
    
    from sklearn.ensemble import RandomForestClassifier
    print("✅ Sklearn imported")
    
    # Test paths
    paralytic_path = r"C:\Users\Aneeta\Downloads\archive (4)\Strokefaces\droopy"
    normal_path = r"C:\Users\Aneeta\Downloads\normal face"
    
    print(f"Paralytic path exists: {os.path.exists(paralytic_path)}")
    print(f"Normal path exists: {os.path.exists(normal_path)}")
    
    if os.path.exists(paralytic_path):
        files = os.listdir(paralytic_path)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"Found {len(image_files)} image files in paralytic folder")
        
        if len(image_files) > 0:
            print(f"First few files: {image_files[:5]}")
    
    print("✅ All imports and path checks successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Test completed!")
