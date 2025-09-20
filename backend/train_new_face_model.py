#!/usr/bin/env python3
"""
Comprehensive Face Classification Training Script
Trains a model to distinguish between paralyzed and normal faces
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceClassificationTrainer:
    def __init__(self, 
                 paralyzed_dir="C:/Users/Aneeta/Downloads/archive (4)/Strokefaces/droopy",
                 normal_dir="C:/Users/Aneeta/Downloads/normal face1/normal face",
                 output_dir="models_new"):
        self.paralyzed_dir = Path(paralyzed_dir)
        self.normal_dir = Path(normal_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Image parameters
        self.image_size = (128, 128)  # Higher resolution for better features
        self.max_images_per_class = 1000  # Balance the dataset
        
        # Models to try
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.training_history = {}
        
    def extract_face_features(self, image):
        """Extract comprehensive features for face classification"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            features = []
            
            # 1. Basic pixel features (sampled for efficiency)
            # Sample every 2nd pixel to reduce dimensionality
            sampled_pixels = gray_array[::2, ::2].flatten()
            features.extend(sampled_pixels)
            
            # 2. Facial Asymmetry Features (Critical for paralysis detection)
            left_half = gray_array[:, :64]
            right_half = gray_array[:, 64:]
            right_flipped = np.fliplr(right_half)
            
            # Multiple asymmetry metrics
            asymmetry_mean = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_std = np.std(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_max = np.max(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            asymmetry_median = np.median(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            
            features.extend([asymmetry_mean, asymmetry_std, asymmetry_max, asymmetry_median])
            
            # 3. Eye Region Analysis (Important for paralysis)
            eye_region_left = gray_array[30:60, 20:60]  # Left eye region
            eye_region_right = gray_array[30:60, 68:108]  # Right eye region
            eye_region_right_flipped = np.fliplr(eye_region_right)
            
            eye_asymmetry_mean = np.mean(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
            eye_asymmetry_std = np.std(np.abs(eye_region_left.astype(float) - eye_region_right_flipped.astype(float)))
            features.extend([eye_asymmetry_mean, eye_asymmetry_std])
            
            # 4. Mouth Region Analysis
            mouth_region_left = gray_array[80:110, 40:80]  # Left mouth region
            mouth_region_right = gray_array[80:110, 48:88]  # Right mouth region
            mouth_region_right_flipped = np.fliplr(mouth_region_right)
            
            mouth_asymmetry_mean = np.mean(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
            mouth_asymmetry_std = np.std(np.abs(mouth_region_left.astype(float) - mouth_region_right_flipped.astype(float)))
            features.extend([mouth_asymmetry_mean, mouth_asymmetry_std])
            
            # 5. Edge Features
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (self.image_size[0] * self.image_size[1])
            features.append(edge_density)
            
            # 6. Texture Features (Local Binary Patterns approximation)
            texture_features = []
            for i in range(0, self.image_size[0], 16):
                for j in range(0, self.image_size[1], 16):
                    block = gray_array[i:i+16, j:j+16]
                    if block.size > 0:
                        texture_features.extend([
                            np.mean(block),
                            np.std(block),
                            np.var(block)
                        ])
            features.extend(texture_features)
            
            # 7. Histogram Features
            hist, _ = np.histogram(gray_array.flatten(), bins=32, range=(0, 256))
            hist_normalized = hist / np.sum(hist)
            features.extend(hist_normalized)
            
            # 8. Statistical Features
            stats_features = [
                np.mean(gray_array),
                np.std(gray_array),
                np.var(gray_array),
                np.median(gray_array),
                np.percentile(gray_array, 25),
                np.percentile(gray_array, 75),
                np.percentile(gray_array, 90),
                np.percentile(gray_array, 95)
            ]
            features.extend(stats_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def load_dataset(self):
        """Load and balance the dataset"""
        logger.info("🔄 Loading dataset...")
        
        X = []
        y = []
        
        # Load paralyzed images
        logger.info(f"Loading paralyzed images from {self.paralyzed_dir}")
        paralyzed_count = 0
        # Search recursively for image files
        paralyzed_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            paralyzed_files.extend(self.paralyzed_dir.rglob(ext))
        
        for img_path in paralyzed_files:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                if paralyzed_count >= self.max_images_per_class:
                    break
                
                try:
                    with Image.open(img_path) as img:
                        features = self.extract_face_features(img)
                        if features is not None:
                            X.append(features)
                            y.append(1)  # 1 for paralyzed
                            paralyzed_count += 1
                            
                    if paralyzed_count % 100 == 0:
                        logger.info(f"  Loaded {paralyzed_count} paralyzed images")
                        
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
        
        logger.info(f"✅ Loaded {paralyzed_count} paralyzed images")
        
        # Load normal images (sample to balance dataset)
        logger.info(f"Loading normal images from {self.normal_dir}")
        normal_count = 0
        # Search recursively for image files
        normal_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            normal_files.extend(self.normal_dir.rglob(ext))
        random.shuffle(normal_files)  # Randomize selection
        
        for img_path in normal_files:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                if normal_count >= self.max_images_per_class:
                    break
                
                try:
                    with Image.open(img_path) as img:
                        features = self.extract_face_features(img)
                        if features is not None:
                            X.append(features)
                            y.append(0)  # 0 for normal
                            normal_count += 1
                            
                    if normal_count % 100 == 0:
                        logger.info(f"  Loaded {normal_count} normal images")
                        
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
        
        logger.info(f"✅ Loaded {normal_count} normal images")
        
        if len(X) == 0:
            raise ValueError("No images loaded! Check your dataset paths.")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"📊 Total dataset: {len(X)} images")
        logger.info(f"   Normal: {sum(y == 0)}")
        logger.info(f"   Paralyzed: {sum(y == 1)}")
        logger.info(f"   Feature count: {X.shape[1]}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        logger.info("🤖 Training models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} images")
        logger.info(f"Test set: {len(X_test)} images")
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = 0
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Calculate class weights for balanced training
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
                
                # Set class weights if model supports it
                if hasattr(model, 'class_weight'):
                    model.class_weight = class_weight_dict
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on test set
                test_score = model.score(X_test_scaled, y_test)
                logger.info(f"{model_name} test accuracy: {test_score:.4f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                logger.info(f"{model_name} CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Store results
                self.training_history[model_name] = {
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Select best model
                if test_score > best_score:
                    best_score = test_score
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info(f"🏆 Best model: {self.best_model_name} with test accuracy: {best_score:.4f}")
        return X_test_scaled, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model"""
        logger.info("📊 Evaluating best model...")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC Score: {auc_score:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Normal', 'Paralyzed']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(cm)
        
        # Save evaluation results
        evaluation_results = {
            'test_accuracy': float(accuracy),
            'roc_auc_score': float(auc_score),
            'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Paralyzed'], output_dict=True),
            'confusion_matrix': cm.tolist(),
            'model_name': self.best_model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def save_model(self):
        """Save the trained model and scaler"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        logger.info("💾 Saving model...")
        
        # Save model
        model_path = self.output_dir / 'face_classification_model.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Save scaler
        scaler_path = self.output_dir / 'face_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save model info
        model_info = {
            'model_type': str(type(self.best_model).__name__),
            'model_name': self.best_model_name,
            'image_size': self.image_size,
            'feature_count': int(self.best_model.n_features_in_),
            'training_date': datetime.now().isoformat(),
            'training_history': self.training_history,
            'class_labels': {0: 'normal', 1: 'paralyzed'}
        }
        
        info_path = self.output_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"✅ Model saved to {model_path}")
        logger.info(f"✅ Scaler saved to {scaler_path}")
        logger.info(f"✅ Model info saved to {info_path}")
        
        return model_path, scaler_path, info_path
    
    def plot_results(self, X_test, y_test):
        """Create visualization plots"""
        try:
            logger.info("📈 Creating visualization plots...")
            
            # Make predictions
            y_pred = self.best_model.predict(X_test)
            y_pred_proba = self.best_model.predict_proba(X_test)
            
            # Create plots directory
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Paralyzed'],
                       yticklabels=['Normal', 'Paralyzed'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Model Comparison
            model_names = list(self.training_history.keys())
            test_scores = [self.training_history[name]['test_score'] for name in model_names]
            cv_means = [self.training_history[name]['cv_mean'] for name in model_names]
            cv_stds = [self.training_history[name]['cv_std'] for name in model_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Test scores
            ax1.bar(model_names, test_scores, color='skyblue', alpha=0.7)
            ax1.set_title('Model Test Scores')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(test_scores):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # CV scores with error bars
            ax2.bar(model_names, cv_means, yerr=cv_stds, color='lightcoral', alpha=0.7, capsize=5)
            ax2.set_title('Model Cross-Validation Scores')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
                ax2.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Plots saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    def train_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting Face Classification Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Load dataset
            X, y = self.load_dataset()
            
            # Train models
            X_test, y_test = self.train_models(X, y)
            
            # Evaluate best model
            evaluation_results = self.evaluate_model(X_test, y_test)
            
            # Save model
            model_path, scaler_path, info_path = self.save_model()
            
            # Create plots
            self.plot_results(X_test, y_test)
            
            logger.info("=" * 60)
            logger.info("✅ Training pipeline completed successfully!")
            logger.info(f"🏆 Best model: {self.best_model_name}")
            logger.info(f"📊 Test accuracy: {evaluation_results['test_accuracy']:.4f}")
            logger.info(f"📊 ROC AUC: {evaluation_results['roc_auc_score']:.4f}")
            logger.info(f"💾 Model saved to: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the training"""
    print("🚀 Starting Face Classification Model Training")
    print("=" * 60)
    
    # Create trainer
    trainer = FaceClassificationTrainer()
    
    # Run training pipeline
    success = trainer.train_complete_pipeline()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Model files saved to: models_new/")
        print("📊 Check models_new/plots/ for training visualizations")
        print("🔧 Use the model with: python predict_face.py <image_path>")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
