#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json

def preprocess_image(image_path, target_size=(32, 32)):
    """Preprocess image for training"""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('L').resize(target_size)
        # Convert to numpy array and flatten
        return np.array(image).flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    print("🚀 Starting Facial Paralysis Model Retraining...")
    print("=" * 60)
    
    # Paths
    paralytic_path = r"C:\Users\Aneeta\Downloads\archive (4)\Strokefaces\droopy"
    normal_path = r"C:\Users\Aneeta\Downloads\normal face"
    
    print("📁 Loading paralytic images...")
    paralytic_features = []
    paralytic_labels = []
    
    if os.path.exists(paralytic_path):
        image_files = [f for f in os.listdir(paralytic_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(image_files)} paralytic images")
        
        # Process first 200 images
        for i, image_file in enumerate(image_files[:200]):
            if i % 25 == 0:
                print(f"  Processing paralytic image {i+1}/200")
            
            image_path = os.path.join(paralytic_path, image_file)
            features = preprocess_image(image_path)
            
            if features is not None:
                paralytic_features.append(features)
                paralytic_labels.append(1)  # 1 for paralytic
    else:
        print(f"❌ Paralytic path not found: {paralytic_path}")
        return
    
    print(f"✅ Loaded {len(paralytic_features)} paralytic images")
    
    print("📁 Loading normal images...")
    normal_features = []
    normal_labels = []
    
    # Extract LFW dataset if needed
    lfw_path = os.path.join(normal_path, "lfw_funneled")
    
    if not os.path.exists(lfw_path):
        print("  Extracting LFW dataset...")
        import tarfile
        lfw_tar = os.path.join(normal_path, "lfw-funneled.tgz")
        with tarfile.open(lfw_tar, 'r:gz') as tar:
            tar.extractall(normal_path)
        print("  ✅ LFW dataset extracted!")
    
    if os.path.exists(lfw_path):
        person_dirs = [d for d in os.listdir(lfw_path) 
                      if os.path.isdir(os.path.join(lfw_path, d))]
        
        print(f"  Found {len(person_dirs)} person directories in LFW")
        
        processed = 0
        for person_dir in person_dirs:
            if processed >= 200:  # Limit to 200 normal images
                break
                
            person_path = os.path.join(lfw_path, person_dir)
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files[:1]:  # 1 per person
                if processed >= 200:
                    break
                    
                if processed % 25 == 0:
                    print(f"  Processing normal image {processed+1}/200")
                
                image_path = os.path.join(person_path, image_file)
                features = preprocess_image(image_path)
                
                if features is not None:
                    normal_features.append(features)
                    normal_labels.append(0)  # 0 for normal
                    processed += 1
    else:
        print(f"❌ LFW path not found: {lfw_path}")
        return
    
    print(f"✅ Loaded {len(normal_features)} normal images")
    
    if len(paralytic_features) == 0 or len(normal_features) == 0:
        print("❌ Error: Could not load sufficient data")
        return
    
    print("🤖 Training model...")
    
    # Combine datasets
    all_features = paralytic_features + normal_features
    all_labels = paralytic_labels + normal_labels
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"  Total dataset: {len(X)} images")
    print(f"  Paralytic: {sum(y)}, Normal: {len(y) - sum(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} images")
    print(f"  Test set: {len(X_test)} images")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Fitting model...")
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Paralytic']))
    
    print("💾 Saving model...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/retrained_ai_model.pkl"
    joblib.dump(model, model_path)
    
    # Save info
    info = {
        "model_type": "RandomForestClassifier",
        "feature_count": int(model.n_features_in_),
        "n_estimators": int(model.n_estimators),
        "accuracy": float(accuracy),
        "training_date": "2025-09-20"
    }
    
    with open("models/model_info_retrained.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Model saved to {model_path}")
    
    print("=" * 60)
    print("✅ Training completed successfully!")
    print(f"📊 Final accuracy: {accuracy:.4f}")
    print(f"💾 Model saved to: {model_path}")

if __name__ == "__main__":
    main()
