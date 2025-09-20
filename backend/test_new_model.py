#!/usr/bin/env python3
"""
Model Testing Script for Facial Paralysis Detection
Tests the trained model with various scenarios
"""

import os
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_dir="models", data_dir="data/new_dataset"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        self.model_info = None
        self.load_model()
        
        # Image parameters
        self.target_size = (64, 64)
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load model
            model_path = self.model_dir / 'ai_model.pkl'
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("✅ Model loaded successfully")
            else:
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Load scaler
            scaler_path = self.model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Scaler loaded successfully")
            else:
                logger.error(f"❌ Scaler file not found: {scaler_path}")
                return False
            
            # Load model info
            info_path = self.model_dir / 'model_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                logger.info("✅ Model info loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
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
    
    def predict_single_image(self, image_path):
        """Predict a single image"""
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not loaded")
            return None
        
        try:
            # Load and preprocess image
            with Image.open(image_path) as img:
                features = self.extract_enhanced_features(img)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Make prediction
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                
                result = {
                    'prediction': 'Paralyzed' if prediction == 1 else 'Normal',
                    'confidence': float(max(probabilities)),
                    'probabilities': {
                        'normal': float(probabilities[0]),
                        'paralyzed': float(probabilities[1])
                    },
                    'is_paralyzed': bool(prediction)
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return None
    
    def test_on_dataset(self, split='test'):
        """Test model on a specific dataset split"""
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not loaded")
            return None
        
        X, y = [], []
        class_mapping = {'paralyzed': 1, 'normal': 0}
        
        # Load test data
        for class_name, label in class_mapping.items():
            class_dir = self.data_dir / split / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            image_files = list(class_dir.glob('*.jpg'))
            logger.info(f"Testing on {len(image_files)} {class_name} images")
            
            for img_file in image_files:
                try:
                    with Image.open(img_file) as img:
                        features = self.extract_enhanced_features(img)
                        X.append(features)
                        y.append(label)
                except Exception as e:
                    logger.error(f"Error loading {img_file}: {e}")
        
        if not X:
            logger.error("No test data loaded")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"\n=== {split.upper()} SET RESULTS ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Total samples: {len(y)}")
        logger.info(f"Paralyzed samples: {np.sum(y == 1)}")
        logger.info(f"Normal samples: {np.sum(y == 0)}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y, y_pred, target_names=['Normal', 'Paralyzed']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(cm)
        
        # Calculate additional metrics
        true_negatives, false_positives, false_negatives, true_positives = cm.ravel()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"\nDetailed Metrics:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall (Sensitivity): {recall:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"F1-Score: {f1_score:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y, y_pred, target_names=['Normal', 'Paralyzed'], output_dict=True)
        }
    
    def test_individual_images(self, test_images_dir):
        """Test model on individual images"""
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
            return
        
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        if not image_files:
            logger.error(f"No images found in {test_dir}")
            return
        
        logger.info(f"\n=== TESTING INDIVIDUAL IMAGES ===")
        logger.info(f"Found {len(image_files)} images to test")
        
        results = []
        for img_file in image_files:
            result = self.predict_single_image(img_file)
            if result:
                results.append({
                    'image': str(img_file),
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
                logger.info(f"{img_file.name}: {result['prediction']} (confidence: {result['confidence']:.4f})")
        
        return results
    
    def create_visualization(self, results):
        """Create visualization of test results"""
        try:
            plots_dir = self.model_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Create confusion matrix heatmap
            if 'confusion_matrix' in results:
                plt.figure(figsize=(8, 6))
                sns.heatmap(results['confusion_matrix'], 
                           annot=True, 
                           fmt='d', 
                           cmap='Blues',
                           xticklabels=['Normal', 'Paralyzed'],
                           yticklabels=['Normal', 'Paralyzed'])
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def run_comprehensive_test(self):
        """Run comprehensive testing"""
        logger.info("Starting comprehensive model testing...")
        
        if not self.load_model():
            logger.error("Failed to load model")
            return False
        
        # Test on different splits
        test_results = {}
        
        for split in ['train', 'validation', 'test']:
            if (self.data_dir / split).exists():
                logger.info(f"\n{'='*50}")
                results = self.test_on_dataset(split)
                if results:
                    test_results[split] = results
        
        # Save test results
        results_file = self.model_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"\nTest results saved to {results_file}")
        
        # Create visualizations
        if test_results:
            self.create_visualization(test_results.get('test', {}))
        
        logger.info("Comprehensive testing completed!")
        return True

def main():
    """Main function to run the testing"""
    # Check if model exists
    model_dir = "models"
    if not Path(model_dir).exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("Please run train_new_model.py first to train the model.")
        return
    
    # Create tester
    tester = ModelTester()
    
    # Run comprehensive testing
    success = tester.run_comprehensive_test()
    
    if success:
        print("✅ Model testing completed successfully!")
        print(f"📊 Check models/plots/ for visualizations")
        print(f"📄 Check models/test_results.json for detailed results")
    else:
        print("❌ Model testing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
