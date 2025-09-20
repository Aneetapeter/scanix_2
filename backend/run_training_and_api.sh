#!/bin/bash
echo "🚀 Starting Face Classification Training and API"
echo "================================================"

echo ""
echo "Step 1: Training the model..."
echo "This will take 10-30 minutes depending on your system"
echo ""
python train_new_face_model.py

echo ""
echo "Step 2: Starting API server..."
echo "API will be available at http://localhost:5001"
echo "Press Ctrl+C to stop the API server"
echo ""
python face_classification_api.py
