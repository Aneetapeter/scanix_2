from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import base64
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Global variables for the model
model = None
model_loaded = False

def load_model():
    """Load the CNN model for facial paralysis detection"""
    global model, model_loaded
    try:
        # For demo purposes, we'll create a simple model
        # In production, you would load a pre-trained model
        model = create_demo_model()
        model_loaded = True
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False

def create_demo_model():
    """Create a demo CNN model for facial paralysis detection"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Initialize weights (in production, load pre-trained weights)
    model.build((None, 224, 224, 3))
    
    return model

def preprocess_image(image_path):
    """Preprocess the uploaded image for model input"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def detect_facial_paralysis(image_path):
    """Detect facial paralysis using the CNN model"""
    global model, model_loaded
    
    if not model_loaded:
        return None
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine if paralysis is detected (threshold = 0.5)
        has_paralysis = confidence > 0.5
        
        # Generate recommendation based on confidence
        if has_paralysis:
            if confidence > 0.8:
                recommendation = "High confidence of facial paralysis detected. Please consult a neurologist immediately for urgent evaluation and treatment."
            else:
                recommendation = "Potential facial paralysis detected. We recommend scheduling an appointment with a neurologist for further evaluation."
        else:
            if confidence < 0.2:
                recommendation = "No signs of facial paralysis detected. Continue regular health monitoring."
            else:
                recommendation = "Low probability of facial paralysis. If you have concerns, consider consulting a healthcare provider."
        
        return {
            'has_paralysis': has_paralysis,
            'confidence': confidence,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error in facial paralysis detection: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for facial paralysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save uploaded file temporarily
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        # Analyze the image
        result = detect_facial_paralysis(file_path)
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        if result is None:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/doctors', methods=['GET'])
def get_doctors():
    """Get list of available doctors"""
    doctors = [
        {
            'id': '1',
            'name': 'Dr. Sarah Johnson',
            'specialization': 'Neurology',
            'image_url': 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=face',
            'rating': 4.9,
            'experience': 15,
            'is_online': True,
            'hospital': 'Mayo Clinic',
            'languages': ['English', 'Spanish'],
            'bio': 'Specialized in facial nerve disorders and stroke rehabilitation.'
        },
        {
            'id': '2',
            'name': 'Dr. Michael Chen',
            'specialization': 'Telemedicine',
            'image_url': 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=face',
            'rating': 4.8,
            'experience': 12,
            'is_online': True,
            'hospital': 'Johns Hopkins',
            'languages': ['English', 'Mandarin'],
            'bio': 'Expert in remote neurological consultations and AI-assisted diagnosis.'
        },
        {
            'id': '3',
            'name': 'Dr. Emily Rodriguez',
            'specialization': 'Neurology',
            'image_url': 'https://images.unsplash.com/photo-1594824373636-4b0b0b0b0b0b?w=300&h=300&fit=crop&crop=face',
            'rating': 4.7,
            'experience': 10,
            'is_online': False,
            'hospital': 'Cleveland Clinic',
            'languages': ['English', 'Spanish', 'Portuguese'],
            'bio': 'Focused on early detection and treatment of facial paralysis.'
        },
        {
            'id': '4',
            'name': 'Dr. James Wilson',
            'specialization': 'Telemedicine',
            'image_url': 'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=300&h=300&fit=crop&crop=face',
            'rating': 4.9,
            'experience': 18,
            'is_online': True,
            'hospital': 'Massachusetts General',
            'languages': ['English'],
            'bio': 'Pioneer in telemedicine and digital health solutions.'
        }
    ]
    
    return jsonify(doctors)

@app.route('/send-report', methods=['POST'])
def send_report_to_doctor():
    """Send analysis report to a doctor"""
    try:
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        result = data.get('result')
        
        if not doctor_id or not result:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # In a real application, you would:
        # 1. Store the report in a database
        # 2. Send notification to the doctor
        # 3. Log the interaction
        
        return jsonify({'message': 'Report sent successfully'})
    
    except Exception as e:
        print(f"Error in send_report_to_doctor: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/contact', methods=['POST'])
def submit_contact_form():
    """Handle contact form submission"""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')
        
        if not all([name, email, message]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # In a real application, you would:
        # 1. Store the message in a database
        # 2. Send email notification
        # 3. Log the submission
        
        return jsonify({'message': 'Contact form submitted successfully'})
    
    except Exception as e:
        print(f"Error in submit_contact_form: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Load the model when the app starts
    load_model()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
