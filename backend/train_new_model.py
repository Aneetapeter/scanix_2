#!/usr/bin/env python3
"""
Model Training Script for Facial Paralysis Detection
Trains a new model using the preprocessed dataset
"""

import os
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialParalysisTrainer:
    def __init__(self, data_dir="data/new_dataset", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Image parameters
        self.target_size = (64, 64)
        
        # Models to try
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
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
    
    def extract_enhanced_features(self, image):
        """Extract enhanced features for facial paralysis detection"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Extract multiple feature types
            features = []
            
            # 1. Basic pixel features
            basic_features = gray_array.flatten()
            features.extend(basic_features)
            
            # 2. Asymmetry features (crucial for paralysis detection)
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
            
            # 4. Texture features
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
            raise Exception(f"Feature extraction error: {e}")
    
    def load_dataset(self, split='train'):
        """Load dataset for a specific split"""
        X, y = [], []
        class_mapping = {'paralyzed': 1, 'normal': 0}
        
        for class_name, label in class_mapping.items():
            class_dir = self.data_dir / split / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            image_files = list(class_dir.glob('*.jpg'))
            logger.info(f"Loading {len(image_files)} images from {class_dir}")
            
            for img_file in image_files:
                try:
                    with Image.open(img_file) as img:
                        features = self.extract_enhanced_features(img)
                        X.append(features)
                        y.append(label)
                except Exception as e:
                    logger.error(f"Error loading {img_file}: {e}")
        
        return np.array(X), np.array(y)
    
    def train_models(self):
        """Train all models and select the best one"""
        logger.info("Loading training data...")
        X_train, y_train = self.load_dataset('train')
        
        logger.info("Loading validation data...")
        X_val, y_val = self.load_dataset('validation')
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        best_score = 0
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation set
                val_score = model.score(X_val_scaled, y_val)
                logger.info(f"{model_name} validation accuracy: {val_score:.4f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                logger.info(f"{model_name} CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Store results
                self.training_history[model_name] = {
                    'validation_score': val_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Select best model
                if val_score > best_score:
                    best_score = val_score
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info(f"Best model: {self.best_model_name} with validation accuracy: {best_score:.4f}")
        return self.best_model, self.best_model_name
    
    def evaluate_model(self):
        """Evaluate the best model on test set"""
        logger.info("Loading test data...")
        X_test, y_test = self.load_dataset('test')
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Normal', 'Paralyzed']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(cm)
        
        # Save evaluation results
        evaluation_results = {
            'test_accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Paralyzed'], output_dict=True),
            'confusion_matrix': cm.tolist(),
            'model_name': self.best_model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.model_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def save_model(self):
        """Save the trained model and scaler"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        # Save model
        model_path = self.model_dir / 'ai_model.pkl'
        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'model_type': str(type(self.best_model).__name__),
            'target_size': self.target_size,
            'feature_count': self.best_model.n_features_in_ if hasattr(self.best_model, 'n_features_in_') else 'Unknown',
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.model_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {self.model_dir / 'model_info.json'}")
    
    def plot_training_results(self):
        """Plot training results"""
        try:
            # Create plots directory
            plots_dir = self.model_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Plot model comparison
            model_names = list(self.training_history.keys())
            val_scores = [self.training_history[name]['validation_score'] for name in model_names]
            cv_means = [self.training_history[name]['cv_mean'] for name in model_names]
            cv_stds = [self.training_history[name]['cv_std'] for name in model_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Validation scores
            ax1.bar(model_names, val_scores, color='skyblue', alpha=0.7)
            ax1.set_title('Model Validation Scores')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(val_scores):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # Cross-validation scores with error bars
            ax2.bar(model_names, cv_means, yerr=cv_stds, color='lightcoral', alpha=0.7, capsize=5)
            ax2.set_title('Model Cross-Validation Scores')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
                ax2.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    def train_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        try:
            # Train models
            self.train_models()
            
            # Evaluate best model
            evaluation_results = self.evaluate_model()
            
            # Save model
            self.save_model()
            
            # Create plots
            self.plot_training_results()
            
            logger.info("Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False

def main():
    """Main function to run the training"""
    # Check if dataset exists
    data_dir = "data/new_dataset"
    if not Path(data_dir).exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        print("Please run prepare_new_dataset.py first to create the dataset.")
        return
    
    # Create trainer
    trainer = FacialParalysisTrainer(data_dir)
    
    # Run training pipeline
    success = trainer.train_complete_pipeline()
    
    if success:
        print("✅ Model training completed successfully!")
        print(f"📁 Model saved to: models/")
        print(f"📊 Check models/plots/ for training visualizations")
    else:
        print("❌ Model training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
