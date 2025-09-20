from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io
import json
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

def extract_working_enhanced_features(image):
    """Extract enhanced features without OpenCV dependency"""
    try:
        # Convert to grayscale
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Resize to target size (64x64)
        img_resized = np.array(gray.resize((64, 64)))
        
        # Extract asymmetry features (key for paralysis detection)
        left_half = img_resized[:, :32]
        right_half = img_resized[:, 32:]
        
        # Flip right half to compare with left
        right_flipped = np.fliplr(right_half)
        
        # Calculate asymmetry metrics
        asymmetry = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
        
        # Extract edge features using simple gradient
        grad_x = np.abs(np.diff(img_resized, axis=1))
        grad_y = np.abs(np.diff(img_resized, axis=0))
        edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (64 * 64)
        
        # Extract texture features using simple filters
        texture_features = []
        
        # Mean and std of different regions
        for i in range(0, 64, 16):
            for j in range(0, 64, 16):
                region = img_resized[i:i+16, j:j+16]
                if region.size > 0:
                    texture_features.extend([
                        np.mean(region),
                        np.std(region),
                        np.var(region)
                    ])
        
        # Combine all features
        basic_features = img_resized.flatten()  # Basic pixel features (4096)
        asymmetry_features = [asymmetry, edge_density]  # 2 features
        combined_features = np.concatenate([basic_features, asymmetry_features, texture_features])
        
        return combined_features
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Fallback to basic features
        return np.array(image.convert('L').resize((64, 64))).flatten()

def preprocess_image(image_bytes):
    """Enhanced preprocessing for facial paralysis detection"""
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract enhanced features
        features = extract_working_enhanced_features(image)
        
        return features.reshape(1, -1)
    except Exception as e:
        raise Exception(f"Image preprocessing error: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'message': 'Working Enhanced AI Backend is running!'
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze image for facial paralysis"""
    try:
        print("\n🔍 Received analysis request")
        print(f"Content-Type: {request.content_type}")
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
                
                # Scale features if scaler is available
                if scaler is not None:
                    img_array = scaler.transform(img_array)
                
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
        
        # Handle JSON data
        elif request.is_json:
            data = request.get_json()
            if 'image_data' in data:
                print("✅ Received JSON image data")
                # Process base64 image data here if needed
                return jsonify({'error': 'Base64 processing not implemented yet'})
        
        return jsonify({'error': 'No valid image data received'})
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_recommendations(prediction, confidence):
    """Generate recommendations based on prediction"""
    if prediction == 1:  # Paralysis detected
        if confidence > 0.8:
            return [
                "High confidence paralysis detected",
                "Seek immediate medical attention",
                "Consult a neurologist or facial nerve specialist",
                "Consider emergency care if symptoms are severe"
            ]
        else:
            return [
                "Possible facial paralysis detected",
                "Schedule a medical consultation",
                "Monitor symptoms closely",
                "Consider seeing a healthcare provider"
            ]
    else:  # Normal
        if confidence > 0.8:
            return [
                "No signs of facial paralysis detected",
                "Continue regular health monitoring",
                "Maintain good facial muscle health",
                "Regular check-ups recommended"
            ]
        else:
            return [
                "Uncertain result - low confidence",
                "Consider retaking the test",
                "Monitor for any facial changes",
                "Consult healthcare provider if concerned"
            ]

@app.route('/doctors', methods=['GET'])
def get_doctors():
    """Get list of recommended doctors"""
    doctors = [
        {
            'id': 1,
            'name': 'Dr. Sarah Johnson',
            'specialty': 'Neurologist',
            'experience': '15 years',
            'rating': 4.8,
            'location': 'New York, NY',
            'phone': '+1 (555) 123-4567'
        },
        {
            'id': 2,
            'name': 'Dr. Michael Chen',
            'specialty': 'Facial Nerve Specialist',
            'experience': '12 years',
            'rating': 4.9,
            'location': 'Los Angeles, CA',
            'phone': '+1 (555) 234-5678'
        },
        {
            'id': 3,
            'name': 'Dr. Emily Rodriguez',
            'specialty': 'Neurologist',
            'experience': '10 years',
            'rating': 4.7,
            'location': 'Chicago, IL',
            'phone': '+1 (555) 345-6789'
        }
    ]
    return jsonify(doctors)

if __name__ == '__main__':
    print("🚀 Starting Working Enhanced AI Server...")
    print("✅ Supports multipart file uploads")
    print("✅ Supports JSON requests")
    print("✅ Enhanced facial paralysis detection")
    print("✅ No OpenCV dependency")
    print("⚠️  Running in debug mode - not for production!")
    print("🌐 Server will be available at:")
    print("   http://127.0.0.1:5000")
    print("   http://localhost:5000")
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)
