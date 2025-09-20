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
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process image
        image = image.convert('RGB').resize((32, 32), Image.Resampling.LANCZOS)
        img_array = np.array(image).flatten().reshape(1, -1)
        
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
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Simple AI Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
