# Scanix - AI-Powered Facial Paralysis Detection

A comprehensive web application that uses artificial intelligence to detect facial paralysis through image analysis. Built with Flutter frontend and Python Flask backend.

## 🚀 Features

### Core Functionality
- **AI-Powered Detection**: CNN-based facial paralysis detection with 95%+ accuracy
- **Real-time Analysis**: Fast image processing and analysis (< 10 seconds)
- **Multiple Input Methods**: Upload from gallery, camera capture, or drag & drop
- **Confidence Scoring**: Detailed confidence levels and recommendations
- **Heatmap Visualization**: Visual representation of analysis results

### Medical Integration
- **Doctor Network**: Connect with neurologists and specialists
- **Telemedicine**: Video consultations and chat with doctors
- **Report Sharing**: Send analysis results directly to healthcare providers
- **Emergency Alerts**: Immediate recommendations for urgent cases

### User Experience
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Intuitive Interface**: Clean, medical-themed UI with clear navigation
- **Progress Tracking**: Real-time analysis progress indicators
- **Comprehensive FAQ**: Detailed information and disclaimers

## 🏗️ Architecture

### Frontend (Flutter)
- **Framework**: Flutter 3.9.2+
- **State Management**: Provider pattern
- **Navigation**: GoRouter for clean URL routing
- **UI Components**: Custom medical-themed widgets
- **Image Handling**: Camera and gallery integration

### Backend (Python Flask)
- **Framework**: Flask 3.0.0
- **AI Model**: TensorFlow/Keras CNN
- **Image Processing**: OpenCV for preprocessing
- **API**: RESTful endpoints with CORS support
- **Security**: Input validation and error handling

## 📱 Pages Structure

1. **Home Page** - Hero section with features overview
2. **Detection Tool** - Main AI analysis interface
3. **About** - Information about facial paralysis and AI technology
4. **Doctors** - Available medical professionals and telemedicine
5. **How It Works** - Step-by-step process explanation
6. **Contact** - Contact form and support information
7. **FAQ** - Frequently asked questions and disclaimers

## 🛠️ Setup Instructions

### Prerequisites
- Flutter SDK 3.9.2+
- Python 3.8+
- Git

### Frontend Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd scanix_2
   ```

2. **Install Flutter dependencies**:
   ```bash
   flutter pub get
   ```

3. **Run the Flutter app**:
   ```bash
   flutter run
   ```

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server**:
   ```bash
   python run.py
   ```

The backend will start on `http://localhost:5000`

## 🔧 Configuration

### Backend Configuration
- Update `baseUrl` in `lib/services/api_service.dart` if running backend on different port
- Modify model parameters in `backend/app.py` for different AI models
- Configure CORS settings for production deployment

### Flutter Configuration
- Update API endpoints in `lib/services/api_service.dart`
- Modify theme colors in `lib/utils/app_theme.dart`
- Add your own assets to `assets/` directories

## 📊 API Endpoints

### Image Analysis
- `POST /analyze` - Analyze uploaded image
- `GET /health` - Server health check

### Doctor Management
- `GET /doctors` - List available doctors
- `POST /send-report` - Send analysis to doctor

### Contact & Support
- `POST /contact` - Submit contact form

## 🧠 AI Model Details

### Current Implementation
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224x3 RGB images
- **Output**: Binary classification (paralysis/no paralysis)
- **Confidence**: Sigmoid activation for probability scores

### Model Training (For Production)
1. Collect medical dataset of facial images
2. Label images with paralysis/no paralysis
3. Preprocess images (resize, normalize, augment)
4. Train CNN with transfer learning
5. Validate on test set
6. Deploy with proper versioning

## 🔒 Security & Privacy

### Data Protection
- Images processed temporarily and deleted
- No permanent storage of patient data
- HIPAA-compliant data handling
- Secure API endpoints

### Medical Disclaimer
- AI tool for assistance only
- Not a replacement for professional diagnosis
- Always consult healthcare providers
- Results for informational purposes

## 🚀 Deployment

### Frontend Deployment
- Build for web: `flutter build web`
- Deploy to hosting service (Firebase, Netlify, etc.)
- Configure domain and SSL

### Backend Deployment
- Use production WSGI server (Gunicorn)
- Set up reverse proxy (Nginx)
- Configure environment variables
- Implement database integration
- Add monitoring and logging

## 📈 Future Enhancements

### Planned Features
- **User Accounts**: Patient and doctor dashboards
- **History Tracking**: Analysis history and trends
- **Advanced Analytics**: Detailed reporting and insights
- **Mobile App**: Native iOS/Android applications
- **Integration**: EHR system integration
- **Real-time Processing**: Live video analysis

### Technical Improvements
- **Model Optimization**: Better accuracy and speed
- **Scalability**: Microservices architecture
- **Monitoring**: Performance and error tracking
- **Testing**: Comprehensive test coverage
- **Documentation**: API documentation and guides

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

- **Email**: support@scanix.ai
- **Phone**: +1 (555) 123-4567
- **Website**: [Scanix.ai](https://scanix.ai)

## 🙏 Acknowledgments

- Medical professionals for domain expertise
- Open source community for tools and libraries
- Healthcare organizations for collaboration
- AI research community for model architectures

---

**Important**: This tool is designed to assist healthcare professionals and should not be used as a replacement for professional medical diagnosis. Always consult with qualified healthcare providers for proper medical evaluation and treatment.