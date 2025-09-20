#!/usr/bin/env python3
"""
Test API Script for Facial Paralysis Detection
Tests the API endpoints with sample images
"""

import requests
import json
import base64
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def test_health(self):
        """Test the health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Health check passed")
                logger.info(f"Model loaded: {data.get('model_loaded', False)}")
                logger.info(f"Scaler loaded: {data.get('scaler_loaded', False)}")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def test_analyze_endpoint(self, image_path):
        """Test the analyze endpoint with an image"""
        try:
            # Encode image
            image_data = self.encode_image(image_path)
            if not image_data:
                return False
            
            # Prepare request data
            data = {
                'image': image_data
            }
            
            # Send request
            response = requests.post(
                f"{self.base_url}/analyze",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Analysis successful for {Path(image_path).name}")
                logger.info(f"Prediction: {result.get('prediction', 'Unknown')}")
                logger.info(f"Confidence: {result.get('confidence', 0):.4f}")
                logger.info(f"Probabilities: {result.get('probabilities', {})}")
                return True
            else:
                logger.error(f"❌ Analysis failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return False
    
    def test_multipart_upload(self, image_path):
        """Test multipart file upload"""
        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                response = requests.post(f"{self.base_url}/analyze", files=files)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Multipart upload successful for {Path(image_path).name}")
                logger.info(f"Prediction: {result.get('prediction', 'Unknown')}")
                logger.info(f"Confidence: {result.get('confidence', 0):.4f}")
                return True
            else:
                logger.error(f"❌ Multipart upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Multipart upload error: {e}")
            return False
    
    def test_doctors_endpoint(self):
        """Test the doctors endpoint"""
        try:
            response = requests.get(f"{self.base_url}/doctors")
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Doctors endpoint working")
                logger.info(f"Found {len(data.get('doctors', []))} doctors")
                return True
            else:
                logger.error(f"❌ Doctors endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Doctors endpoint error: {e}")
            return False
    
    def run_comprehensive_test(self, test_images_dir=None):
        """Run comprehensive API testing"""
        logger.info("🧪 Starting comprehensive API testing...")
        
        # Test health endpoint
        if not self.test_health():
            logger.error("❌ Health check failed. Make sure the server is running.")
            return False
        
        # Test doctors endpoint
        self.test_doctors_endpoint()
        
        # Test with sample images if provided
        if test_images_dir:
            test_dir = Path(test_images_dir)
            if test_dir.exists():
                image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
                
                if image_files:
                    logger.info(f"\n📸 Testing with {len(image_files)} images...")
                    
                    for img_file in image_files[:3]:  # Test first 3 images
                        logger.info(f"\nTesting image: {img_file.name}")
                        
                        # Test JSON upload
                        self.test_analyze_endpoint(img_file)
                        
                        # Test multipart upload
                        self.test_multipart_upload(img_file)
                else:
                    logger.warning("No test images found in the specified directory")
            else:
                logger.warning(f"Test images directory not found: {test_images_dir}")
        else:
            logger.info("No test images directory provided. Skipping image tests.")
        
        logger.info("\n🎉 API testing completed!")
        return True

def main():
    """Main function to run API testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Facial Paralysis Detection API')
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--images', help='Directory containing test images')
    
    args = parser.parse_args()
    
    # Create tester
    tester = APITester(args.url)
    
    # Run comprehensive test
    success = tester.run_comprehensive_test(args.images)
    
    if success:
        print("✅ API testing completed successfully!")
    else:
        print("❌ API testing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
