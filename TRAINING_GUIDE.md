# 🧠 Scanix AI Training Guide

## Overview
Your AI model needs to be trained on your actual dataset to properly detect facial paralysis. This guide will help you train the model using your real data.

## 🎯 What We're Training
- **Input**: Your facial paralysis images (from `backend/data/raw_data/Strokefaces/droopy/`)
- **Output**: A trained AI model that can detect facial paralysis
- **Goal**: High accuracy in distinguishing between normal and paralyzed faces

## 📋 Prerequisites
1. **Python 3.8+** installed
2. **Your dataset** in `backend/data/raw_data/Strokefaces/droopy/`
3. **Required packages** (will be installed automatically)

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Run Training
```bash
python run_training.py
```

## 📊 What Happens During Training

### 1. **Dataset Preparation**
- ✅ Loads your paralysis images from `droopy/` folder
- ✅ Creates synthetic normal faces to balance the dataset
- ✅ Splits data into train/validation/test sets (70%/15%/15%)

### 2. **Model Training**
- ✅ Creates an advanced CNN model
- ✅ Trains on your actual data
- ✅ Uses data augmentation to improve accuracy
- ✅ Monitors training progress

### 3. **Model Evaluation**
- ✅ Tests the model on unseen data
- ✅ Shows accuracy, precision, recall metrics
- ✅ Generates confusion matrix

### 4. **Model Saving**
- ✅ Saves the trained model to `models/` folder
- ✅ Creates model metadata
- ✅ Generates training history plots

## 📈 Expected Results

After training, you should see:
- **Accuracy**: 85-95% (depending on your data quality)
- **Precision**: High accuracy in detecting paralysis
- **Recall**: Good at finding all paralysis cases
- **Confusion Matrix**: Shows true/false positives/negatives

## 🔧 Troubleshooting

### If Training Fails:
1. **Check your data**: Make sure images are in `backend/data/raw_data/Strokefaces/droopy/`
2. **Install packages**: Run `pip install -r requirements.txt`
3. **Check Python version**: Make sure you have Python 3.8+

### If Accuracy is Low:
1. **More data**: Add more images to your dataset
2. **Better quality**: Ensure images are clear and well-lit
3. **More epochs**: Increase training time in the script

## 📁 File Structure After Training

```
backend/
├── models/
│   ├── facial_paralysis_model/     # Your trained model
│   ├── model_info.json            # Model metadata
│   └── best_facial_paralysis_model.h5  # Best model checkpoint
├── data/
│   └── complete_dataset/          # Processed training data
│       ├── train/
│       ├── validation/
│       └── test/
└── training_history.png           # Training progress plot
```

## 🎉 After Training

Once training is complete:
1. **Your AI model** will be saved in `models/` folder
2. **The backend** will automatically use your trained model
3. **Accuracy** should be much higher than before
4. **Real detection** of facial paralysis based on your data

## 🔄 Re-training

If you get more data or want to improve accuracy:
1. Add new images to your dataset
2. Run the training script again
3. The model will be updated with new data

## 📞 Support

If you encounter any issues:
1. Check the error messages
2. Make sure all dependencies are installed
3. Verify your dataset is in the correct location
4. Check that you have enough images (at least 100+ recommended)

---

**Happy Training! 🚀**

Your AI will be much more accurate after training on your actual data!
