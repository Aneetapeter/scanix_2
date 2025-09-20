import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset():
    print("🔍 ANALYZING FACIAL PARALYSIS DATASETS")
    print("=" * 50)
    
    # Dataset paths
    paralytic_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'archive (4)', 'Strokefaces', 'droopy')
    normal_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'normal face')
    
    print(f"📁 Paralytic dataset path: {paralytic_path}")
    print(f"📁 Normal dataset path: {normal_path}")
    print()
    
    # Check if paths exist
    paralytic_exists = os.path.exists(paralytic_path)
    normal_exists = os.path.exists(normal_path)
    
    print(f"✅ Paralytic dataset exists: {paralytic_exists}")
    print(f"✅ Normal dataset exists: {normal_exists}")
    print()
    
    if not paralytic_exists or not normal_exists:
        print("❌ One or both datasets not found!")
        return
    
    # Count files
    paralytic_files = [f for f in os.listdir(paralytic_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    normal_files = [f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📊 Paralytic images: {len(paralytic_files)}")
    print(f"📊 Normal images: {len(normal_files)}")
    print()
    
    # Analyze sample images
    print("🔬 ANALYZING SAMPLE IMAGES")
    print("-" * 30)
    
    # Analyze paralytic images
    print("Paralytic face characteristics:")
    paralytic_samples = paralytic_files[:5]
    for i, filename in enumerate(paralytic_samples):
        try:
            img_path = os.path.join(paralytic_path, filename)
            img = Image.open(img_path)
            print(f"  Sample {i+1}: {filename}")
            print(f"    Size: {img.size}")
            print(f"    Mode: {img.mode}")
            print(f"    Format: {img.format}")
        except Exception as e:
            print(f"    Error loading {filename}: {e}")
    print()
    
    # Analyze normal images
    print("Normal face characteristics:")
    normal_samples = normal_files[:5]
    for i, filename in enumerate(normal_samples):
        try:
            img_path = os.path.join(normal_path, filename)
            img = Image.open(img_path)
            print(f"  Sample {i+1}: {filename}")
            print(f"    Size: {img.size}")
            print(f"    Mode: {img.mode}")
            print(f"    Format: {img.format}")
        except Exception as e:
            print(f"    Error loading {filename}: {e}")
    print()
    
    # Check for LFW dataset
    lfw_tgz_path = os.path.join(normal_path, 'lfw-funneled.tgz')
    lfw_extracted_path = os.path.join(normal_path, 'lfw_funneled')
    
    print("🔍 CHECKING LFW DATASET")
    print("-" * 25)
    print(f"LFW TGZ exists: {os.path.exists(lfw_tgz_path)}")
    print(f"LFW extracted exists: {os.path.exists(lfw_extracted_path)}")
    
    if os.path.exists(lfw_extracted_path):
        lfw_dirs = [d for d in os.listdir(lfw_extracted_path) if os.path.isdir(os.path.join(lfw_extracted_path, d))]
        print(f"LFW person directories: {len(lfw_dirs)}")
        
        # Count LFW images
        lfw_images = 0
        for person_dir in lfw_dirs[:10]:  # Check first 10 directories
            person_path = os.path.join(lfw_extracted_path, person_dir)
            person_images = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            lfw_images += len(person_images)
            print(f"  {person_dir}: {len(person_images)} images")
        
        print(f"Total LFW images (first 10 people): {lfw_images}")
    
    print()
    print("🎯 RECOMMENDATIONS")
    print("-" * 20)
    print("1. Use LFW dataset for normal faces (high quality, diverse)")
    print("2. Focus on facial asymmetry features for paralysis detection")
    print("3. Use higher resolution images (64x64 or 128x128)")
    print("4. Apply face detection and alignment before training")
    print("5. Use data augmentation to increase dataset size")

if __name__ == '__main__':
    analyze_dataset()
