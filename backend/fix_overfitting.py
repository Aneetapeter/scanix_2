#!/usr/bin/env python3
"""
Quick fix for overfitting - Create a simpler, more robust model
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from datetime import datetime

class SimpleRobustTrainer:
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.image_size = (64, 64)  # Keep same size as current model
        self.scaler = StandardScaler()
        
    def extract_simple_features(self, image_path):
        """Extract simple, robust features"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                
                # Convert to grayscale
                gray = img.convert('L')
                gray_array = np.array(gray)
                
                features = []
                
                # 1. Basic pixel features (all pixels)
                basic_features = gray_array.flatten()
                features.extend(basic_features)
                
                # 2. Asymmetry features (most important for paralysis)
                left_half = gray_array[:, :32]
                right_half = gray_array[:, 32:]
                right_flipped = np.fliplr(right_half)
                
                # Calculate asymmetry metrics
                asymmetry = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
                features.append(asymmetry)
                
                # 3. Edge features
                grad_x = np.abs(np.diff(gray_array, axis=1))
                grad_y = np.abs(np.diff(gray_array, axis=0))
                edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (64 * 64)
                features.append(edge_density)
                
                # 4. Texture features (block-based)
                for i in range(0, 64, 8):
                    for j in range(0, 64, 8):
                        block = gray_array[i:i+8, j:j+8]
                        if block.size > 0:
                            features.extend([np.mean(block), np.std(block)])
                            if block.shape[0] > 1 and block.shape[1] > 1:
                                grad_x = np.abs(np.diff(block, axis=1))
                                grad_y = np.abs(np.diff(block, axis=0))
                                features.extend([np.mean(grad_x), np.mean(grad_y)])
                
                # 5. Statistical features
                stats_features = [
                    np.mean(gray_array),
                    np.std(gray_array),
                    np.var(gray_array),
                    np.median(gray_array),
                    np.percentile(gray_array, 25),
                    np.percentile(gray_array, 75)
                ]
                features.extend(stats_features)
                
                return np.array(features)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset(self, max_images_per_class=200):
        """Load dataset with balanced classes"""
        print("🚀 Loading dataset...")
        
        X = []
        y = []
        
        # Load normal images
        normal_dir = self.data_dir / 'train' / 'normal'
        if normal_dir.exists():
            print(f"Loading normal images from {normal_dir}")
            normal_count = 0
            for img_path in normal_dir.glob('*.jpg'):
                if normal_count >= max_images_per_class:
                    break
                    
                features = self.extract_simple_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(0)  # 0 for normal
                    normal_count += 1
                    
                if normal_count % 50 == 0:
                    print(f"  Processed normal image {normal_count}")
            
            print(f"✅ Loaded {normal_count} normal images")
        else:
            print(f"❌ Normal images directory not found: {normal_dir}")
        
        # Load paralysis images
        paralysis_dir = self.data_dir / 'train' / 'paralysis'
        if paralysis_dir.exists():
            print(f"Loading paralysis images from {paralysis_dir}")
            paralysis_count = 0
            for img_path in paralysis_dir.glob('*.jpg'):
                if paralysis_count >= max_images_per_class:
                    break
                    
                features = self.extract_simple_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(1)  # 1 for paralysis
                    paralysis_count += 1
                    
                if paralysis_count % 50 == 0:
                    print(f"  Processed paralysis image {paralysis_count}")
            
            print(f"✅ Loaded {paralysis_count} paralysis images")
        else:
            print(f"❌ Paralysis images directory not found: {paralysis_dir}")
        
        if len(X) == 0:
            raise ValueError("No images found! Check your data directory structure.")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Total dataset: {len(X)} images")
        print(f"   Normal: {sum(y == 0)}")
        print(f"   Paralysis: {sum(y == 1)}")
        
        return X, y
    
    def train_simple_model(self, X, y):
        """Train a simple, robust model"""
        print("🤖 Training simple robust model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for balanced training
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Train simple Random Forest with regularization
        model = RandomForestClassifier(
            n_estimators=100,      # Reduced for simplicity
            max_depth=10,          # Reduced for better generalization
            min_samples_split=10,  # Increased for regularization
            min_samples_leaf=5,    # Increased for regularization
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight_dict,
            bootstrap=True,
            oob_score=True
        )
        
        print("Fitting simple model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Model accuracy: {accuracy:.4f}")
        print(f"📊 Out-of-bag score: {model.oob_score_:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Paralysis']))
        
        # Cross-validation
        print("\nCross-validation scores:")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, accuracy, X_test_scaled, y_test, y_pred
    
    def save_simple_model(self, model, accuracy, X_test, y_test):
        """Save the simple model and metadata"""
        print("💾 Saving simple model...")
        
        # Save model
        model_path = self.models_dir / 'ai_model.pkl'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save model info
        model_info = {
            'model_type': 'RandomForestClassifier',
            'image_size': self.image_size,
            'feature_count': int(model.n_features_in_),
            'n_estimators': int(model.n_estimators),
            'max_depth': int(model.max_depth),
            'accuracy': float(accuracy),
            'oob_score': float(model.oob_score_),
            'training_date': datetime.now().isoformat(),
            'total_images': len(X_test) * 5,  # Approximate
            'normal_images': int(sum(y_test == 0) * 5),
            'paralysis_images': int(sum(y_test == 1) * 5),
            'improvements': [
                'Simplified model to reduce overfitting',
                'Regularized parameters for better generalization',
                'Balanced class weights',
                'Robust feature extraction',
                'Cross-validation for validation'
            ]
        }
        
        info_path = self.models_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
        print(f"✅ Model info saved to {info_path}")
        
        return model_path, scaler_path, info_path
    
    def run_simple_training(self, max_images_per_class=200):
        """Run the complete simple training process"""
        print("🚀 Starting Simple Robust Training...")
        print("=" * 50)
        
        try:
            # Load dataset
            X, y = self.load_dataset(max_images_per_class)
            
            # Train simple model
            model, accuracy, X_test, y_test, y_pred = self.train_simple_model(X, y)
            
            # Save model
            model_path, scaler_path, info_path = self.save_simple_model(model, accuracy, X_test, y_test)
            
            print("=" * 50)
            print("✅ Simple training completed successfully!")
            print(f"📊 Final accuracy: {accuracy:.4f}")
            print(f"📊 Out-of-bag score: {model.oob_score_:.4f}")
            print(f"💾 Model saved to: {model_path}")
            
            return model, accuracy
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

def main():
    """Main function for command line usage"""
    print("Starting simple robust training...")
    trainer = SimpleRobustTrainer()
    model, accuracy = trainer.run_simple_training()
    
    if model is not None:
        print(f"\n🎉 Success! Your simple robust model is ready!")
        print(f"📊 Accuracy: {accuracy:.2%}")
        print("🔧 Key improvements:")
        print("  - Simplified model to reduce overfitting")
        print("  - Regularized parameters")
        print("  - Better generalization")
        print("  - Robust feature extraction")
    else:
        print("\n❌ Training failed. Check your data directory and try again.")

if __name__ == "__main__":
    main()
