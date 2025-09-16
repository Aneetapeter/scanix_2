# Scanix Backend - AI Facial Paralysis Detection

This is the Python Flask backend for the Scanix AI-powered facial paralysis detection application.

## Features

- **AI-Powered Analysis**: CNN model for facial paralysis detection
- **Image Processing**: OpenCV-based image preprocessing
- **RESTful API**: Flask-based API endpoints
- **Doctor Integration**: Telemedicine and doctor management
- **Contact System**: Contact form handling

## Setup

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask Server**:
   ```bash
   python app.py
   ```

3. **Access the API**:
   - Base URL: `http://localhost:5000`
   - Health Check: `http://localhost:5000/health`

## API Endpoints

### Image Analysis
- **POST** `/analyze` - Analyze uploaded image for facial paralysis
  - Input: Image file (multipart/form-data)
  - Output: Analysis results with confidence score and recommendations

### Doctors
- **GET** `/doctors` - Get list of available doctors
  - Output: Array of doctor objects with details

### Reports
- **POST** `/send-report` - Send analysis report to doctor
  - Input: JSON with doctor_id and result data
  - Output: Success confirmation

### Contact
- **POST** `/contact` - Submit contact form
  - Input: JSON with name, email, and message
  - Output: Success confirmation

### Health
- **GET** `/health` - Health check endpoint
  - Output: Server status and model loading status

## Model Information

The current implementation uses a demo CNN model. In production, you would:

1. Train a model on real medical data
2. Use transfer learning with pre-trained models
3. Implement proper model versioning
4. Add model performance monitoring

## Security Considerations

- Input validation for all endpoints
- File upload security
- CORS configuration
- Error handling and logging

## Development Notes

- The model is loaded once at startup
- Temporary files are cleaned up after processing
- All responses include proper error handling
- CORS is enabled for Flutter frontend integration

## Production Deployment

For production deployment:

1. Use a production WSGI server (Gunicorn)
2. Set up proper logging
3. Configure environment variables
4. Use a reverse proxy (Nginx)
5. Implement database integration
6. Add authentication and authorization
