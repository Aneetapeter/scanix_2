print("Starting simple test...")

import os
print("OS imported")

import numpy as np
print("NumPy imported")

from PIL import Image
print("PIL imported")

print("Testing image processing...")

paralytic_path = r"C:\Users\Aneeta\Downloads\archive (4)\Strokefaces\droopy"

if os.path.exists(paralytic_path):
    files = os.listdir(paralytic_path)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) > 0:
        test_file = os.path.join(paralytic_path, image_files[0])
        print(f"Testing with file: {test_file}")
        
        try:
            image = Image.open(test_file).convert('L').resize((32, 32))
            features = np.array(image).flatten()
            print(f"Successfully processed image. Features shape: {features.shape}")
        except Exception as e:
            print(f"Error processing image: {e}")
else:
    print("Paralytic path not found")

print("Test completed!")
