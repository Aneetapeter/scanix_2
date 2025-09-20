#!/usr/bin/env python3
"""
Dataset Extraction Script
Extracts the normal face dataset from the zip file
"""

import zipfile
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_zip_file(zip_path, extract_to):
    """Extract zip file to specified directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✅ Successfully extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Error extracting {zip_path}: {e}")
        return False

def main():
    """Main function to extract the dataset"""
    zip_file = r"C:\Users\Aneeta\Downloads\normal face1.zip"
    extract_to = r"C:\Users\Aneeta\Downloads"
    
    # Check if zip file exists
    if not Path(zip_file).exists():
        logger.error(f"❌ Zip file not found: {zip_file}")
        return False
    
    # Check if already extracted
    extracted_path = Path(extract_to) / "normal face1"
    if extracted_path.exists():
        logger.info(f"✅ Dataset already extracted to: {extracted_path}")
        return True
    
    # Extract the zip file
    logger.info(f"Extracting {zip_file} to {extract_to}...")
    success = extract_zip_file(zip_file, extract_to)
    
    if success:
        logger.info("✅ Dataset extraction completed!")
        logger.info(f"📁 Extracted to: {extracted_path}")
        return True
    else:
        logger.error("❌ Dataset extraction failed!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Dataset extraction completed successfully!")
        print("You can now run the training pipeline with: python run_complete_pipeline.py")
    else:
        print("\n❌ Dataset extraction failed. Please check the error messages above.")
