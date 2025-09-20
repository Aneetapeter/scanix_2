#!/usr/bin/env python3
"""
Quick AI Training for Facial Paralysis Detection
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def main():
    print("Starting Quick AI Training...")
    
    # Setup
    data_dir = Path('data')
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Load images
    print("Loading images...")
    X = []
    y = []
    
    # Load normal images
    normal_dir = data_dir / 'train' / 'normal'
    if normal_dir.exists():
        for img_path in normal_dir.glob('*.jpg'):
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((32, 32), Image.Resampling.LANCZOS)
                    X.append(np.array(img).flatten())
                    y.append(0)  # normal
            except:
                continue
    
    # Load paralysis images
    paralysis_dir = data_dir / 'train' / 'paralysis'
    if paralysis_dir.exists():
        for img_path in paralysis_dir.glob('*.jpg'):
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((32, 32), Image.Resampling.LANCZOS)
                    X.append(np.array(img).flatten())
                    y.append(1)  # paralysis
            except:
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} images")
    print(f"Normal: {sum(y == 0)}, Paralysis: {sum(y == 1)}")
    
    if len(X) == 0:
        print("No images found!")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training AI model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"AI Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Paralysis']))
    
    # Save model
    joblib.dump(model, 'models/ai_model.pkl')
    
    # Save info
    info = {
        'accuracy': float(accuracy),
        'total_images': len(X),
        'normal_images': int(sum(y == 0)),
        'paralysis_images': int(sum(y == 1)),
        'model_type': 'RandomForest'
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ AI Training Complete!")
    print("Model saved to: models/ai_model.pkl")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
