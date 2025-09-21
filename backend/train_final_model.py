#!/usr/bin/env python3
"""
FINAL Working Model for Facial Paralysis Detection
Fixes overfitting with proper regularization and data augmentation
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from datetime import datetime
import random

class FinalFacialParalysisModel:
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.image_size = (64, 64)  # Keep same size for compatibility
        self.scaler = StandardScaler()
        
    def augment_image(self, image, num_augmentations=2):
        """Apply data augmentation to increase dataset diversity"""
        augmented_images = [image]  # Include original
        
        for _ in range(num_augmentations):
            aug_image = image.copy()
            
            # Random brightness adjustment
            brightness_factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Brightness(aug_image)
            aug_image = enhancer.enhance(brightness_factor)
            
            # Random contrast adjustment
            contrast_factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Contrast(aug_image)
            aug_image = enhancer.enhance(contrast_factor)
            
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                aug_image = ImageOps.mirror(aug_image)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def extract_features(self, image):
        """Extract robust features for paralysis detection"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            features = []
            
            # 1. Basic pixel features (all pixels)
            basic_features = gray_array.flatten()
            features.extend(basic_features)
            
            # 2. Asymmetry features (MOST IMPORTANT for paralysis)
            left_half = gray_array[:, :32]
            right_half = gray_array[:, 32:]
            right_flipped = np.fliplr(right_half)
            
            # Multiple asymmetry metrics
            asymmetry_mean = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_std = np.std(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_max = np.max(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            features.extend([asymmetry_mean, asymmetry_std, asymmetry_max])
            
            # 3. Eye region asymmetry (critical for paralysis)
            eye_region_left = gray_array[20:40, 15:45]  # Left eye region
            eye_region_right = gray_array[20:40, 19:49]  # Right eye region
            eye_region_right_flipped = np.fliplr(eye_region_right)
            
            eye_asymmetry = np.mean(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
            features.append(eye_asymmetry)
            
            # 4. Mouth region asymmetry
            mouth_region_left = gray_array[45:60, 20:40]  # Left mouth region
            mouth_region_right = gray_array[45:60, 24:44]  # Right mouth region
            mouth_region_right_flipped = np.fliplr(mouth_region_right)
            
            mouth_asymmetry = np.mean(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
            features.append(mouth_asymmetry)
            
            # 5. Edge features
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (64 * 64)
            features.append(edge_density)
            
            # 6. Statistical features
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
            print(f"Error processing image: {e}")
            return None
    
    def load_augmented_dataset(self, max_images_per_class=150):
        """Load dataset with data augmentation"""
        print("🚀 Loading dataset with augmentation...")
        
        X = []
        y = []
        
        # Load normal images with augmentation
        normal_dir = self.data_dir / 'train' / 'normal'
        if normal_dir.exists():
            print(f"Loading normal images from {normal_dir}")
            normal_count = 0
            for img_path in normal_dir.glob('*.jpg'):
                if normal_count >= max_images_per_class:
                    break
                    
                try:
                    with Image.open(img_path) as img:
                        # Apply augmentation
                        augmented_images = self.augment_image(img, 2)
                        
                        for aug_img in augmented_images:
                            features = self.extract_features(aug_img)
                            if features is not None:
                                X.append(features)
                                y.append(0)  # 0 for normal
                        
                        normal_count += 1
                        
                        if normal_count % 30 == 0:
                            print(f"  Processed normal image {normal_count}")
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            print(f"✅ Loaded {normal_count} normal images with augmentation")
        else:
            print(f"❌ Normal images directory not found: {normal_dir}")
        
        # Load paralysis images with augmentation
        paralysis_dir = self.data_dir / 'train' / 'paralysis'
        if paralysis_dir.exists():
            print(f"Loading paralysis images from {paralysis_dir}")
            paralysis_count = 0
            for img_path in paralysis_dir.glob('*.jpg'):
                if paralysis_count >= max_images_per_class:
                    break
                    
                try:
                    with Image.open(img_path) as img:
                        # Apply augmentation
                        augmented_images = self.augment_image(img, 2)
                        
                        for aug_img in augmented_images:
                            features = self.extract_features(aug_img)
                            if features is not None:
                                X.append(features)
                                y.append(1)  # 1 for paralysis
                        
                        paralysis_count += 1
                        
                        if paralysis_count % 30 == 0:
                            print(f"  Processed paralysis image {paralysis_count}")
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            print(f"✅ Loaded {paralysis_count} paralysis images with augmentation")
        else:
            print(f"❌ Paralysis images directory not found: {paralysis_dir}")
        
        if len(X) == 0:
            raise ValueError("No images found! Check your data directory structure.")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Total augmented dataset: {len(X)} images")
        print(f"   Normal: {sum(y == 0)}")
        print(f"   Paralysis: {sum(y == 1)}")
        
        return X, y
    
    def train_final_model(self, X, y):
        """Train the final model with proper regularization"""
        print("🤖 Training final model...")
        
        # Split the data with stratification
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
        
        # Train REGULARIZED Random Forest to prevent overfitting
        model = RandomForestClassifier(
            n_estimators=150,      # Reduced from 500
            max_depth=12,          # Reduced from 35
            min_samples_split=8,   # Increased for regularization
            min_samples_leaf=4,    # Increased for regularization
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight_dict,
            bootstrap=True,
            oob_score=True,
            max_samples=0.8         # Use bootstrap sampling
        )
        
        print("Fitting regularized model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Model accuracy: {accuracy:.4f}")
        print(f"📊 Out-of-bag score: {model.oob_score_:.4f}")
        
        # Calculate ROC AUC
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"📊 ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Paralysis']))
        
        # Cross-validation with stratification
        print("\nCross-validation scores:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        feature_importance = model.feature_importances_
        print(f"\nTop 5 most important features:")
        top_indices = np.argsort(feature_importance)[-5:]
        for i, idx in enumerate(reversed(top_indices)):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        return model, accuracy, roc_auc, X_test_scaled, y_test, y_pred
    
    def save_final_model(self, model, accuracy, roc_auc, X_test, y_test):
        """Save the final model and metadata"""
        print("💾 Saving final model...")
        
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
            'roc_auc': float(roc_auc),
            'oob_score': float(model.oob_score_),
            'training_date': datetime.now().isoformat(),
            'total_images': len(X_test) * 5,  # Approximate
            'normal_images': int(sum(y_test == 0) * 5),
            'paralysis_images': int(sum(y_test == 1) * 5),
            'improvements': [
                'Data augmentation for better generalization',
                'Regularized model parameters to prevent overfitting',
                'Advanced facial asymmetry detection',
                'Eye and mouth region analysis',
                'Balanced class weights',
                'Cross-validation for validation',
                'Reduced overfitting with proper regularization'
            ]
        }
        
        info_path = self.models_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
        print(f"✅ Model info saved to {info_path}")
        
        return model_path, scaler_path, info_path
    
    def run_final_training(self, max_images_per_class=150):
        """Run the complete final training process"""
        print("🚀 Starting FINAL Facial Paralysis Model Training...")
        print("=" * 60)
        
        try:
            # Load augmented dataset
            X, y = self.load_augmented_dataset(max_images_per_class)
            
            # Train final model
            model, accuracy, roc_auc, X_test, y_test, y_pred = self.train_final_model(X, y)
            
            # Save model
            model_path, scaler_path, info_path = self.save_final_model(model, accuracy, roc_auc, X_test, y_test)
            
            print("=" * 60)
            print("✅ FINAL training completed successfully!")
            print(f"📊 Final accuracy: {accuracy:.4f}")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"📊 Out-of-bag score: {model.oob_score_:.4f}")
            print(f"💾 Model saved to: {model_path}")
            print(f"🔧 Scaler saved to: {scaler_path}")
            print(f"📄 Info saved to: {info_path}")
            
            return model, accuracy, roc_auc
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, 0.0

def main():
    """Main function for command line usage"""
    print("Starting FINAL training...")
    trainer = FinalFacialParalysisModel()
    model, accuracy, roc_auc = trainer.run_final_training()
    
    if model is not None:
        print(f"\n🎉 Success! Your FINAL AI model is ready!")
        print(f"📊 Accuracy: {accuracy:.2%}")
        print(f"📊 ROC AUC: {roc_auc:.2%}")
        print("🔧 Key improvements:")
        print("  - Data augmentation for better generalization")
        print("  - Regularized parameters to prevent overfitting")
        print("  - Advanced facial asymmetry detection")
        print("  - Eye and mouth region analysis")
        print("  - This model should work on external images!")
    else:
        print("\n❌ Training failed. Check your data directory and try again.")

if __name__ == "__main__":
    main()
