# Facial Paralysis Detection - Training Guide

This guide will help you train a new model for detecting paralyzed vs normal faces using your datasets.

## Prerequisites

1. **Python 3.8+** installed
2. **Required packages** installed (see requirements.txt)
3. **Your datasets** in the specified locations:
   - Paralyzed faces: `C:\Users\Aneeta\Downloads\archive (4)\Strokefaces\droopy`
   - Normal faces: `C:\Users\Aneeta\Downloads\normal face`

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
cd backend
python run_complete_pipeline.py
```

This will automatically:
1. Preprocess your datasets
2. Train a new model
3. Test the model
4. Save everything for use in the backend

### Option 2: Run Steps Individually

#### Step 1: Preprocess Data
```bash
cd backend
python prepare_new_dataset.py
```

This will:
- Copy images from your source directories
- Resize all images to 64x64 pixels
- Split data into train/validation/test sets (70%/15%/15%)
- Save processed data to `data/new_dataset/`

#### Step 2: Train Model
```bash
python train_new_model.py
```

This will:
- Train multiple models (Random Forest, SVM, Logistic Regression)
- Select the best performing model
- Save the trained model and scaler to `models/`
- Create training visualizations

#### Step 3: Test Model
```bash
python test_new_model.py
```

This will:
- Test the model on all dataset splits
- Generate detailed performance metrics
- Create confusion matrix and other visualizations
- Save test results to `models/test_results.json`

## Understanding the Results

### Model Performance
After training, you'll see results like:
```
Best model: random_forest with validation accuracy: 0.8542
Test Accuracy: 0.8234
```

### Key Metrics
- **Accuracy**: Overall correctness
- **Precision**: How many predicted paralyzed cases were actually paralyzed
- **Recall**: How many actual paralyzed cases were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### Files Created
- `models/ai_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/model_info.json` - Model metadata
- `models/evaluation_results.json` - Detailed evaluation
- `models/test_results.json` - Test results
- `models/plots/` - Training visualizations

## Using the Trained Model

### Start the Backend Server
```bash
python app.py
```

The server will automatically load your new trained model.

### Test the API
```bash
python test_api.py --images data/new_dataset/test
```

### API Endpoints
- `GET /health` - Check if model is loaded
- `POST /analyze` - Analyze an image for paralysis
- `GET /doctors` - Get list of available doctors

## Troubleshooting

### Common Issues

1. **"Dataset directory not found"**
   - Make sure your image paths are correct
   - Check that the directories contain image files

2. **"No images were processed successfully"**
   - Check image file formats (supports .jpg, .jpeg, .png)
   - Ensure images are not corrupted

3. **"Model training failed"**
   - Check that you have enough data (recommended: 100+ images per class)
   - Ensure images are properly preprocessed

4. **"API not responding"**
   - Make sure the backend server is running
   - Check that the model files exist in `models/`

### Performance Tips

1. **More Data = Better Performance**
   - Aim for at least 200+ images per class
   - Ensure balanced dataset (similar number of paralyzed and normal images)

2. **Image Quality**
   - Use clear, well-lit face images
   - Avoid heavily cropped or distorted images
   - Ensure faces are centered and visible

3. **Model Selection**
   - Random Forest often works well for this task
   - SVM can be good with smaller datasets
   - Try different models to see what works best for your data

## Advanced Usage

### Customizing the Training

Edit `train_new_model.py` to:
- Change model parameters
- Add new feature extraction methods
- Modify the train/validation/test split ratios

### Adding New Features

Edit the `extract_enhanced_features` function in both training and testing scripts to:
- Add new image features
- Modify existing feature extraction
- Experiment with different image sizes

### Model Evaluation

Check `models/test_results.json` for detailed performance metrics:
- Per-class accuracy
- Confusion matrix
- Classification report
- Cross-validation scores

## Support

If you encounter issues:
1. Check the log messages for specific error details
2. Verify your dataset paths and image formats
3. Ensure all required packages are installed
4. Check that you have sufficient disk space for the processed data

## Next Steps

After successful training:
1. Test the model with new images
2. Integrate with your Flutter app
3. Deploy to production
4. Monitor model performance over time
5. Retrain with additional data as needed
