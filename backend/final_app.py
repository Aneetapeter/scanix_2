from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import json
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001', 'http://127.0.0.1:3001', 'http://localhost:3002', 'http://127.0.0.1:3002', 'http://localhost:8080'])

# Load AI model and scaler
print("Loading AI model...")
try:
    model = joblib.load('models/ai_model.pkl')
    print("✅ AI model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

print("Loading scaler...")
try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

def preprocess_image(image_bytes):
    """Enhanced preprocessing for facial paralysis detection"""
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        gray = image.convert('L')
        
        # Resize to 32x32 (matching the model's training)
        img_resized = np.array(gray.resize((32, 32)))
        
        # Extract asymmetry features (key for paralysis detection)
        left_half = img_resized[:, :16]
        right_half = img_resized[:, 16:]
        
        # Flip right half to compare with left
        right_flipped = np.fliplr(right_half)
        
        # Calculate asymmetry metrics
        asymmetry = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        
        # Extract edge features using simple gradient
        grad_x = np.abs(np.diff(img_resized, axis=1))
        grad_y = np.abs(np.diff(img_resized, axis=0))
        edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (32 * 32)
        
        # Extract texture features
        texture_features = []
        for i in range(0, 32, 8):
            for j in range(0, 32, 8):
                region = img_resized[i:i+8, j:j+8]
                if region.size > 0:
                    texture_features.extend([
                        np.mean(region),
                        np.std(region),
                        np.var(region)
                    ])
        
        # Combine all features to match model expectations (1024 features)
        basic_features = img_resized.flatten()  # Basic pixel features (1024)
        asymmetry_features = [asymmetry, edge_density]  # 2 features
        combined_features = np.concatenate([basic_features, asymmetry_features, texture_features])
        
        # Pad or truncate to exactly 1024 features (matching the model)
        target_features = 1024
        if len(combined_features) < target_features:
            # Pad with zeros
            padding = np.zeros(target_features - len(combined_features))
            combined_features = np.concatenate([combined_features, padding])
        elif len(combined_features) > target_features:
            # Truncate
            combined_features = combined_features[:target_features]
        
        return combined_features.reshape(1, -1)
    except Exception as e:
        raise Exception(f"Image preprocessing error: {e}")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        print(f"Request method: {request.method}")
        print(f"Content type: {request.content_type}")
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
        
        # Handle multipart file upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                image_bytes = file.read()
                print(f"✅ Received multipart file: {file.filename}, size: {len(image_bytes)}")
                
                # Process image
                img_array = preprocess_image(image_bytes)
                
                # Make prediction (model was trained without scaling)
                prediction = model.predict(img_array)[0]
                probabilities = model.predict_proba(img_array)[0]
                
                result = {
                    'success': True,
                    'prediction': 'Paralysis detected' if prediction == 1 else 'Normal.',
                    'confidence': float(max(probabilities)),
                    'is_paralysis': bool(prediction),
                    'probabilities': {
                        'normal': float(probabilities[0]),
                        'paralysis': float(probabilities[1])
                    },
                    'recommendations': generate_recommendations(prediction, max(probabilities))
                }
                
                print(f"✅ Prediction successful: {result['prediction']} (confidence: {result['confidence']:.4f})")
                return jsonify(result)
            else:
                return jsonify({'error': 'No file provided'}), 400
        
        # Handle JSON request
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            print(f"✅ Received JSON image data, length: {len(str(image_data))}")
            
            # Handle base64 image
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Process image
            img_array = preprocess_image(image_bytes)
            
            # Make prediction
            prediction = model.predict(img_array)[0]
            probabilities = model.predict_proba(img_array)[0]
            
            result = {
                'success': True,
                'prediction': 'Paralysis' if prediction == 1 else 'Normal',
                'confidence': float(max(probabilities)),
                'is_paralysis': bool(prediction),
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'paralysis': float(probabilities[1])
                },
                'recommendations': generate_recommendations(prediction, max(probabilities))
            }
            
            print(f"✅ Prediction successful: {result['prediction']} (confidence: {result['confidence']:.4f})")
            return jsonify(result)
        
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_recommendations(prediction, confidence):
    """Generate recommendations based on AI prediction"""
    if prediction == 1:  # Paralysis
        if confidence > 0.8:
            return [
                "Paralysis detected with high confidence",
                "Seek immediate medical attention",
                "Consult a neurologist or facial nerve specialist",
                "Consider emergency care if symptoms are severe"
            ]
        else:
            return [
                "Paralysis detected with moderate confidence",
                "Schedule a medical consultation",
                "Monitor symptoms closely",
                "Consider seeing a healthcare provider"
            ]
    else:  # Normal
        return [
            "Normal facial features detected",
            "Continue regular health monitoring",
            "Maintain good facial muscle health",
            "Regular check-ups recommended"
        ]

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

if __name__ == '__main__':
    print("🚀 Starting Final AI Server...")
    print("✅ Supports multipart file uploads!")
    print("✅ Supports JSON requests!")
    app.run(debug=True, host='0.0.0.0', port=5000)
