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
CORS(app)

# Load AI model
print("Loading AI model...")
try:
    model = joblib.load('models/ai_model.pkl')
    print("✅ AI model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def preprocess_image(image_data):
    """Preprocess image for AI prediction"""
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
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 32x32 (same as training)
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and flatten
        img_array = np.array(image).flatten()
        
        # Ensure the array has the right shape
        if len(img_array) != 3072:  # 32*32*3
            raise Exception(f"Invalid image size: {len(img_array)}, expected 3072")
        
        return img_array.reshape(1, -1)
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
        
        # Handle both multipart and JSON requests
        image_data = None
        
        # Check if it's a multipart request (file upload)
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                image_data = file.read()
                print(f"Received multipart file: {file.filename}, size: {len(image_data)}")
        
        # Check if it's JSON request
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            print(f"Received JSON image data, length: {len(str(image_data))}")
        
        # Check if it's form data
        elif 'image' in request.form:
            image_data = request.form['image']
            print(f"Received form image data, length: {len(str(image_data))}")
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process image
        img_array = preprocess_image(image_data)
        
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
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        
        print(f"Report sent to doctor {doctor_id}: {result}")
        
        return jsonify({
            'success': True,
            'message': f'Report sent to doctor {doctor_id}',
            'timestamp': '2025-09-20T02:00:00Z'
        })
    
    except Exception as e:
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
        
        print(f"Contact form: {name} ({email}) - {message}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your message. We will get back to you soon!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Fixed AI Server...")
    print("Supports both multipart and JSON requests!")
    app.run(debug=True, host='0.0.0.0', port=5000)
