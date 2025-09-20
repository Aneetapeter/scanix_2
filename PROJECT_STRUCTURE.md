# Scanix 2 - Facial Paralysis Detection App

## Project Overview
A Flutter mobile application with Python backend for detecting facial paralysis using machine learning.

## Cleaned Project Structure

```
scanix_2/
├── 📱 Flutter App (Frontend)
│   ├── lib/
│   │   ├── main.dart                    # App entry point
│   │   ├── models/                      # Data models
│   │   │   ├── detection_result.dart    # AI detection result model
│   │   │   ├── doctor.dart             # Doctor information model
│   │   │   └── user.dart               # User model
│   │   ├── screens/                     # App screens
│   │   │   ├── modern_home_screen.dart  # Main home screen
│   │   │   ├── enhanced_detection_screen.dart  # AI detection screen
│   │   │   ├── about_screen.dart       # About page
│   │   │   ├── doctors_screen.dart     # Find doctors screen
│   │   │   ├── how_it_works_screen.dart # How it works
│   │   │   ├── contact_screen.dart     # Contact page
│   │   │   ├── faq_screen.dart         # FAQ page
│   │   │   ├── login_screen.dart       # Login screen
│   │   │   ├── signup_screen.dart      # Signup screen
│   │   │   └── camera_screen.dart      # Camera capture screen
│   │   ├── services/                    # Business logic services
│   │   │   ├── ai_analysis_service.dart # AI analysis service
│   │   │   ├── api_service.dart        # API communication
│   │   │   ├── auth_service.dart       # Authentication
│   │   │   ├── demo_service.dart       # Demo data service
│   │   │   └── image_service.dart      # Image processing
│   │   ├── utils/
│   │   │   └── app_theme.dart          # App theming
│   │   └── widgets/                     # Reusable UI components
│   │       ├── app_wrapper.dart
│   │       ├── custom_text_field.dart
│   │       ├── doctor_card.dart
│   │       ├── features_section.dart
│   │       ├── footer_section.dart
│   │       ├── hero_section.dart
│   │       ├── image_upload_widget.dart
│   │       ├── loading_button.dart
│   │       ├── navigation_bar.dart
│   │       ├── professional_card.dart
│   │       ├── professional_loading.dart
│   │       ├── progress_indicator_widget.dart
│   │       ├── result_display_widget.dart
│   │       └── testimonials_section.dart
│   ├── assets/                          # App assets
│   │   ├── animations/                  # Animation files
│   │   ├── icons/                       # App icons
│   │   └── images/                      # Image assets
│   ├── android/                         # Android platform files
│   ├── ios/                            # iOS platform files
│   ├── linux/                          # Linux platform files
│   ├── macos/                          # macOS platform files
│   ├── web/                            # Web platform files
│   ├── windows/                        # Windows platform files
│   ├── test/                           # Unit tests
│   │   └── widget_test.dart
│   ├── pubspec.yaml                    # Flutter dependencies
│   ├── pubspec.lock                    # Locked dependency versions
│   ├── analysis_options.yaml          # Dart analysis options
│   ├── devtools_options.yaml          # DevTools options
│   ├── start_scanix.bat               # Windows startup script
│   └── start_scanix.sh                # Unix startup script
│
├── 🐍 Python Backend (AI/ML)
│   ├── backend/
│   │   ├── inference_service.py        # Main AI inference service
│   │   ├── final_train_model.py       # Model training script
│   │   ├── app.py                     # Flask application
│   │   ├── requirements.txt           # Python dependencies
│   │   ├── models/                    # Trained ML models
│   │   │   ├── ai_model.pkl          # RandomForest model
│   │   │   ├── scaler.pkl            # Feature scaler
│   │   │   └── model_info.json       # Model metadata
│   │   ├── data/                      # Training and test data
│   │   │   ├── dataset_info.json     # Dataset metadata
│   │   │   ├── processed_data/       # Processed training data
│   │   │   │   ├── normal/           # Normal face images
│   │   │   │   └── paralysis/        # Paralysis face images
│   │   │   ├── raw_data/             # Original raw data
│   │   │   │   ├── lfw_funneled/     # LFW normal faces dataset
│   │   │   │   └── Strokefaces/      # Stroke paralysis faces
│   │   │   ├── train/                # Training set
│   │   │   ├── validation/           # Validation set
│   │   │   └── test/                 # Test set
│   │   ├── quick_test.py             # Quick API testing
│   │   └── test_current_model.py     # Model performance testing
│
├── 📚 Documentation
│   ├── README.md                      # Project documentation
│   ├── AI_TRAINING_GUIDE.md          # AI training guide
│   └── PROJECT_STRUCTURE.md          # This file
│
└── 🔧 Configuration
    └── .gitignore                     # Git ignore rules
```

## Key Features

### Flutter Frontend
- **Modern UI**: Clean, responsive design with Material Design
- **Camera Integration**: Capture images directly in the app
- **AI Analysis**: Real-time facial paralysis detection
- **Doctor Finder**: Locate nearby medical professionals
- **User Authentication**: Login/signup functionality
- **Multi-platform**: Android, iOS, Web, Windows, macOS, Linux

### Python Backend
- **Machine Learning**: RandomForest classifier (93.4% accuracy)
- **Feature Extraction**: Advanced facial asymmetry analysis
- **REST API**: Flask-based inference service
- **Model Training**: Comprehensive training pipeline
- **Data Processing**: Automated dataset preparation

## Removed Files (Cleanup Summary)

### Duplicate/Unused Python Files
- `simple_inference_service.py` → Replaced by `inference_service.py`
- `robust_inference_service.py` → Merged into main service
- `advanced_train_model.py` → Replaced by `final_train_model.py`
- `improved_train_model.py` → Duplicate of final trainer
- `train_model.py` → Original version, replaced
- `improved_simple_trainer.py` → Duplicate trainer
- `hybrid_trainer.py` → Alternative approach, not used
- `enhanced_cnn_trainer.py` → CNN approach, not used
- `quick_fix_train.py` → Temporary fix script
- `test_advanced.py` → Basic test, replaced
- `test_ai_service.py` → Duplicate test

### Duplicate/Unused Flutter Files
- `enhanced_home_screen.dart` → Duplicate home screen
- `home_screen.dart` → Duplicate home screen

### Build Artifacts & Cache
- `build/` directory (Flutter build output)
- `backend/__pycache__/` (Python cache)
- `backend/uploads/` (Temporary uploads)
- `backend/archive (4).zip` (Dataset archive)

## Getting Started

### Prerequisites
- Flutter SDK
- Python 3.8+
- Git

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python inference_service.py
```

### Frontend Setup
```bash
flutter pub get
flutter run
```

## File Organization Principles

1. **Single Responsibility**: Each file has one clear purpose
2. **No Duplicates**: Removed all duplicate and backup files
3. **Clean Structure**: Logical grouping of related files
4. **Proper Gitignore**: Comprehensive ignore rules for both Flutter and Python
5. **Documentation**: Clear documentation of project structure and purpose

## Next Steps

1. Test the cleaned project to ensure everything works
2. Commit the cleaned structure to version control
3. Update any documentation that references removed files
4. Consider adding more comprehensive tests
5. Set up CI/CD pipeline for automated testing and deployment
