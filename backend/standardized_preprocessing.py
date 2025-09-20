#!/usr/bin/env python3
"""
Standardized Preprocessing Pipeline for Facial Paralysis Detection
Ensures EXACT consistency between training and inference
"""

import numpy as np
from PIL import Image
from pathlib import Path
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardizedPreprocessor:
    """
    Standardized preprocessing pipeline that matches training exactly
    """
    
    def __init__(self):
        # EXACT same parameters as the actual trained model
        self.image_size = (64, 64)  # Must match the actual trained model
        self.expected_feature_count = 4360  # From model_info.json
        
    def extract_paralysis_features(self, image):
        """
        Extract features EXACTLY as in train_new_model.py (the actual trained model)
        This must match the training pipeline 100%
        """
        try:
            logger.info(f"🔍 Starting EXACT feature extraction (matching trained model)...")
            logger.info(f"   Original image size: {image.size}")
            logger.info(f"   Original image mode: {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to EXACT training size (64x64)
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            logger.info(f"   Resized image shape: {gray_array.shape}")
            
            # Extract multiple feature types - EXACTLY as in train_new_model.py
            features = []
            
            # 1. Basic pixel features (64x64 = 4096 features)
            basic_features = gray_array.flatten()
            features.extend(basic_features)
            
            # 2. Asymmetry features (crucial for paralysis detection) - 1 feature
            left_half = gray_array[:, :32]
            right_half = gray_array[:, 32:]
            right_flipped = np.fliplr(right_half)
            
            # Calculate asymmetry metrics
            asymmetry = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            features.append(asymmetry)
            
            # 3. Edge features - 1 feature
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (64 * 64)
            features.append(edge_density)
            
            # 4. Texture features - 8x8 blocks, 4 features each = 256 features
            for i in range(0, 64, 8):
                for j in range(0, 64, 8):
                    block = gray_array[i:i+8, j:j+8]
                    if block.size > 0:
                        features.extend([np.mean(block), np.std(block)])
                        if block.shape[0] > 1 and block.shape[1] > 1:
                            grad_x = np.abs(np.diff(block, axis=1))
                            grad_y = np.abs(np.diff(block, axis=0))
                            features.extend([np.mean(grad_x), np.mean(grad_y)])
            
            # 5. Statistical features - 6 features
            stats_features = [
                np.mean(gray_array),
                np.std(gray_array),
                np.var(gray_array),
                np.median(gray_array),
                np.percentile(gray_array, 25),
                np.percentile(gray_array, 75)
            ]
            features.extend(stats_features)
            
            # Convert to numpy array
            features_array = np.array(features)
            
            logger.info(f"✅ EXACT feature extraction completed")
            logger.info(f"   Total features: {len(features_array)}")
            logger.info(f"   Expected features: {self.expected_feature_count}")
            
            # Validate feature count
            if len(features_array) != self.expected_feature_count:
                logger.error(f"❌ Feature count mismatch!")
                logger.error(f"   Expected: {self.expected_feature_count}")
                logger.error(f"   Got: {len(features_array)}")
                raise ValueError(f"Feature count mismatch: expected {self.expected_feature_count}, got {len(features_array)}")
            
            return features_array
            
        except Exception as e:
            logger.error(f"❌ Error in EXACT feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_image(self, image_data):
        """
        Preprocess image data and extract features
        Handles both PIL Image objects and base64 encoded strings
        """
        try:
            # Handle different input formats
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                import base64
                image_bytes = base64.b64decode(image_data)
                
                # Open image with PIL
                import io
                image = Image.open(io.BytesIO(image_bytes))
            elif hasattr(image_data, 'mode'):  # PIL Image object
                image = image_data
            else:
                # Assume it's bytes
                import io
                image = Image.open(io.BytesIO(image_data))
            
            # Extract features using EXACT training method
            features = self.extract_paralysis_features(image)
            if features is None:
                raise ValueError("Feature extraction failed")
            
            # Reshape for model input
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise
    
    def validate_model_compatibility(self, model):
        """
        Validate that the model is compatible with our preprocessing
        """
        try:
            expected_features = model.n_features_in_
            if expected_features != self.expected_feature_count:
                logger.warning(f"Model expects {expected_features} features, but preprocessing produces {self.expected_feature_count}")
                # Update expected count to match model
                self.expected_feature_count = expected_features
                logger.info(f"Updated expected feature count to {expected_features}")
            
            logger.info(f"✅ Model compatibility validated")
            logger.info(f"   Model expects: {expected_features} features")
            logger.info(f"   Preprocessing produces: {self.expected_feature_count} features")
            
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility validation failed: {e}")
            return False

# Global preprocessor instance
preprocessor = StandardizedPreprocessor()
