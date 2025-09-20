#!/usr/bin/env python3
"""
Data Preprocessing Script for Facial Paralysis Detection
Handles the new datasets: paralyzed faces and normal faces
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    def __init__(self, paralyzed_path, normal_path, output_dir="data/new_dataset"):
        self.paralyzed_path = Path(paralyzed_path)
        self.normal_path = Path(normal_path)
        self.output_dir = Path(output_dir)
        self.target_size = (64, 64)  # Consistent with existing model
        
        # Create output directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.output_dir / "raw_data" / "paralyzed",
            self.output_dir / "raw_data" / "normal",
            self.output_dir / "processed_data" / "paralyzed",
            self.output_dir / "processed_data" / "normal",
            self.output_dir / "train" / "paralyzed",
            self.output_dir / "train" / "normal",
            self.output_dir / "validation" / "paralyzed",
            self.output_dir / "validation" / "normal",
            self.output_dir / "test" / "paralyzed",
            self.output_dir / "test" / "normal"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def preprocess_image(self, image_path, output_path):
        """Preprocess a single image"""
        try:
            # Open image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target size
                img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
                # Save processed image
                img_resized.save(output_path, 'JPEG', quality=95)
                return True
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def copy_and_preprocess_dataset(self, source_dir, target_class, class_name):
        """Copy and preprocess images for a specific class"""
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_path}")
            return 0
        
        # Check if there's a subdirectory with images (like lfw_funneled)
        subdirs = [d for d in source_path.iterdir() if d.is_dir()]
        if subdirs and not any(f.suffix.lower() in {'.jpg', '.jpeg', '.png'} for f in source_path.iterdir() if f.is_file()):
            # Use the first subdirectory if no images in root
            source_path = subdirs[0]
            logger.info(f"Using subdirectory: {source_path}")
        
        processed_count = 0
        raw_count = 0
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG'}
        image_files = [f for f in source_path.rglob('*') if f.suffix.lower() in image_extensions]
        
        logger.info(f"Found {len(image_files)} images in {source_path}")
        
        for img_file in image_files:
            try:
                # Copy to raw data directory
                raw_dest = self.output_dir / "raw_data" / class_name / img_file.name
                shutil.copy2(img_file, raw_dest)
                raw_count += 1
                
                # Preprocess and save to processed data directory
                processed_dest = self.output_dir / "processed_data" / class_name / f"processed_{img_file.stem}.jpg"
                if self.preprocess_image(img_file, processed_dest):
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
        
        logger.info(f"Successfully processed {processed_count}/{len(image_files)} images for {class_name}")
        return processed_count
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train, validation, and test sets"""
        if not (abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6):
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        for class_name in ['paralyzed', 'normal']:
            processed_dir = self.output_dir / "processed_data" / class_name
            image_files = list(processed_dir.glob('*.jpg'))
            
            if not image_files:
                logger.warning(f"No processed images found for {class_name}")
                continue
            
            # Shuffle images
            random.shuffle(image_files)
            
            # Calculate split indices
            n_images = len(image_files)
            train_end = int(n_images * train_ratio)
            val_end = train_end + int(n_images * val_ratio)
            
            # Split images
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # Copy files to respective directories
            for files, split_name in [(train_files, 'train'), (val_files, 'validation'), (test_files, 'test')]:
                for img_file in files:
                    dest = self.output_dir / split_name / class_name / img_file.name
                    shutil.copy2(img_file, dest)
            
            logger.info(f"{class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    def create_dataset_info(self):
        """Create dataset information file"""
        info = {
            "dataset_name": "Facial Paralysis Detection Dataset",
            "description": "Dataset for detecting paralyzed vs normal faces",
            "classes": {
                "paralyzed": "Images of people with facial paralysis",
                "normal": "Images of people with normal facial expressions"
            },
            "image_size": self.target_size,
            "total_images": 0,
            "split_distribution": {}
        }
        
        # Count images in each split
        for split in ['train', 'validation', 'test']:
            split_info = {}
            for class_name in ['paralyzed', 'normal']:
                count = len(list((self.output_dir / split / class_name).glob('*.jpg')))
                split_info[class_name] = count
                info["total_images"] += count
            info["split_distribution"][split] = split_info
        
        # Save dataset info
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset info saved to {info_file}")
        return info
    
    def process_all(self):
        """Process the complete dataset"""
        logger.info("Starting dataset preprocessing...")
        
        # Process paralyzed faces
        logger.info("Processing paralyzed faces...")
        paralyzed_count = self.copy_and_preprocess_dataset(
            self.paralyzed_path, 
            "paralyzed", 
            "paralyzed"
        )
        
        # Process normal faces
        logger.info("Processing normal faces...")
        normal_count = self.copy_and_preprocess_dataset(
            self.normal_path, 
            "normal", 
            "normal"
        )
        
        if paralyzed_count == 0 or normal_count == 0:
            logger.error("No images were processed successfully. Please check your input paths.")
            return False
        
        # Split dataset
        logger.info("Splitting dataset...")
        self.split_dataset()
        
        # Create dataset info
        info = self.create_dataset_info()
        
        logger.info("Dataset preprocessing completed!")
        logger.info(f"Total images processed: {info['total_images']}")
        logger.info(f"Paralyzed: {paralyzed_count}, Normal: {normal_count}")
        
        return True

def main():
    """Main function to run the preprocessing"""
    # Define paths
    paralyzed_path = r"C:\Users\Aneeta\Downloads\archive (4)\Strokefaces\droopy"
    normal_path = r"C:\Users\Aneeta\Downloads\normal face1\normal face\lfw-funneled"
    output_dir = "data/new_dataset"
    
    # Create preprocessor
    preprocessor = DatasetPreprocessor(paralyzed_path, normal_path, output_dir)
    
    # Process the dataset
    success = preprocessor.process_all()
    
    if success:
        print("✅ Dataset preprocessing completed successfully!")
        print(f"📁 Processed dataset saved to: {output_dir}")
    else:
        print("❌ Dataset preprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
