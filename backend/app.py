#!/usr/bin/env python3
"""
Unified Flask App for Facial Paralysis Detection
Combines the best features from all app versions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import json
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:3000', 'http://127.0.0.1:3000',
    'http://localhost:3001', 'http://127.0.0.1:3001',
    'http://localhost:3002', 'http://127.0.0.1:3002',
    'http://localhost:8080', 'http://127.0.0.1:8080'
])

# Load AI model and scaler
print("Loading AI model...")
try:
    model = joblib.load('models/ai_model.pkl')
    print("✅ AI model loaded successfully!")
    logger.info("AI model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    logger.error(f"Error loading model: {e}")
    model = None

print("Loading scaler...")
try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded successfully!")
    logger.info("Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    logger.warning(f"Scaler not found or error loading: {e}")
    scaler = None

def extract_enhanced_features(image):
    """Extract enhanced features matching the improved training"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 64x64 (matching improved training)
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        
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
        texture_features = []
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

def preprocess_image(image_data):
    """Enhanced preprocessing for facial paralysis detection"""
    try:
        # Handle different input formats
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract enhanced features
        features = extract_enhanced_features(image)
        
        return features.reshape(1, -1)
    except Exception as e:
        raise Exception(f"Image preprocessing error: {e}")

def generate_recommendations(prediction, confidence):
    """Generate recommendations based on AI prediction"""
    if prediction == 1:  # Paralysis
        if confidence > 0.8:
            return [
                "High confidence of facial paralysis detected",
                "Immediate medical consultation recommended",
                "Consider emergency care if symptoms are severe",
                "Document symptoms and seek specialist evaluation"
            ]
        else:
            return [
                "Possible facial paralysis detected",
                "Medical evaluation recommended",
                "Monitor for additional symptoms",
                "Consult with healthcare provider"
            ]
    else:  # Normal
        return [
            "No facial paralysis detected",
            "Continue regular health monitoring",
            "Consult healthcare provider if symptoms develop"
        ]

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'version': '2.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze image for facial paralysis"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content type: {request.content_type}")
        
        # Handle different request types
        image_data = None
        
        # Check if it's a multipart request (file upload)
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                image_data = file.read()
                logger.info(f"Received multipart file: {file.filename}, size: {len(image_data)}")
        
        # Check if it's JSON request
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            logger.info(f"Received JSON image data, length: {len(str(image_data))}")
        
        # Check if it's form data
        elif 'image' in request.form:
            image_data = request.form['image']
            logger.info(f"Received form image data, length: {len(str(image_data))}")
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process image
        img_array = preprocess_image(image_data)
        
        # Scale features if scaler is available
        if scaler is not None:
            img_array = scaler.transform(img_array)
        
        # Make prediction
        prediction = model.predict(img_array)[0]
        probabilities = model.predict_proba(img_array)[0]
        
        result = {
            'success': True,
            'prediction': 'Paralysis detected' if prediction == 1 else 'Normal',
            'confidence': float(max(probabilities)),
            'is_paralysis': bool(prediction),
            'probabilities': {
                'normal': float(probabilities[0]),
                'paralysis': float(probabilities[1])
            },
            'recommendations': generate_recommendations(prediction, max(probabilities))
        }
        
        logger.info(f"Prediction successful: {result['prediction']} (confidence: {result['confidence']:.4f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/doctors', methods=['GET'])
def get_doctors():
    """Get list of available doctors"""
    doctors = [
        {
            'id': 1,
            'name': 'Dr. Sarah Johnson',
            'specialty': 'Neurologist',
            'experience': '15 years',
            'rating': 4.8,
            'available': True,
            'phone': '+1-555-0123',
            'email': 'sarah.johnson@medical.com'
        },
        {
            'id': 2,
            'name': 'Dr. Michael Chen',
            'specialty': 'Facial Plastic Surgeon',
            'experience': '12 years',
            'rating': 4.9,
            'available': True,
            'phone': '+1-555-0124',
            'email': 'michael.chen@medical.com'
        },
        {
            'id': 3,
            'name': 'Dr. Emily Rodriguez',
            'specialty': 'Emergency Medicine',
            'experience': '10 years',
            'rating': 4.7,
            'available': False,
            'phone': '+1-555-0125',
            'email': 'emily.rodriguez@medical.com'
        }
    ]
    
    return jsonify({
        'success': True,
        'doctors': doctors
    })

@app.route('/send-report', methods=['POST'])
def send_report():
    """Send analysis report to doctor"""
    try:
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        result = data.get('result')
        
        if not doctor_id or not result:
            return jsonify({'error': 'Missing required fields'}), 400
        
        logger.info(f"Report sent to doctor {doctor_id}: {result}")
        
        return jsonify({
            'success': True,
            'message': f'Report sent to doctor {doctor_id}',
            'timestamp': '2025-01-20T02:00:00Z'
        })
    
    except Exception as e:
        logger.error(f"Send report error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/contact', methods=['POST'])
def contact():
    """Handle contact form submission"""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')
        
        if not all([name, email, message]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        logger.info(f"Contact form: {name} ({email}) - {message}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your message. We will get back to you soon!'
        })
    
    except Exception as e:
        logger.error(f"Contact form error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Scanix AI Server...")
    print("✅ Supports multipart file uploads!")
    print("✅ Supports JSON requests!")
    print("✅ Enhanced error handling and logging!")
    print("🌐 Server running on: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)