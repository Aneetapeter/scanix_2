import os
import numpy as np
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json
from pathlib import Path

class SimpleFacialParalysisTrainer:
    def __init__(self):
        self.paralytic_path = r"C:\Users\Aneeta\Downloads\archive (4)\Strokefaces\droopy"
        self.normal_path = r"C:\Users\Aneeta\Downloads\normal face"
        self.model_path = "models/retrained_ai_model.pkl"
        self.image_size = (32, 32)  # Smaller size for faster processing
        
    def extract_lfw_dataset(self):
        """Extract LFW dataset if not already extracted"""
        lfw_tar = os.path.join(self.normal_path, "lfw-funneled.tgz")
        lfw_extracted = os.path.join(self.normal_path, "lfw_funneled")
        
        if not os.path.exists(lfw_extracted):
            print("Extracting LFW dataset...")
            import tarfile
            with tarfile.open(lfw_tar, 'r:gz') as tar:
                tar.extractall(self.normal_path)
            print("LFW dataset extracted successfully!")
        else:
            print("LFW dataset already extracted.")
            
        return lfw_extracted
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for training using PIL only"""
        try:
            # Load image with PIL
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.image_size)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Convert to numpy array and flatten
            img_array = np.array(image)
            features = img_array.flatten()
            
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_paralytic_images(self, max_images=500):
        """Load and preprocess paralytic face images"""
        print("Loading paralytic face images...")
        paralytic_features = []
        paralytic_labels = []
        
        if not os.path.exists(self.paralytic_path):
            print(f"Paralytic path not found: {self.paralytic_path}")
            return [], []
        
        image_files = [f for f in os.listdir(self.paralytic_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(image_files)} paralytic images")
        
        processed = 0
        for image_file in image_files:
            if processed >= max_images:
                break
                
            if processed % 50 == 0:
                print(f"Processing paralytic image {processed+1}/{min(max_images, len(image_files))}")
                
            image_path = os.path.join(self.paralytic_path, image_file)
            features = self.preprocess_image(image_path)
            
            if features is not None:
                paralytic_features.append(features)
                paralytic_labels.append(1)  # 1 for paralytic
                processed += 1
        
        print(f"Successfully processed {len(paralytic_features)} paralytic images")
        return paralytic_features, paralytic_labels
    
    def load_normal_images(self, max_images=500):
        """Load and preprocess normal face images from LFW dataset"""
        print("Loading normal face images...")
        normal_features = []
        normal_labels = []
        
        lfw_extracted = self.extract_lfw_dataset()
        
        if not os.path.exists(lfw_extracted):
            print(f"LFW extracted path not found: {lfw_extracted}")
            return [], []
        
        # Get all subdirectories (person names)
        person_dirs = [d for d in os.listdir(lfw_extracted) 
                      if os.path.isdir(os.path.join(lfw_extracted, d))]
        
        print(f"Found {len(person_dirs)} person directories in LFW")
        
        processed_count = 0
        for person_dir in person_dirs:
            if processed_count >= max_images:
                break
                
            person_path = os.path.join(lfw_extracted, person_dir)
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Take up to 3 images per person to avoid bias
            for image_file in image_files[:3]:
                if processed_count >= max_images:
                    break
                    
                if processed_count % 50 == 0:
                    print(f"Processing normal image {processed_count+1}/{max_images}")
                
                image_path = os.path.join(person_path, image_file)
                features = self.preprocess_image(image_path)
                
                if features is not None:
                    normal_features.append(features)
                    normal_labels.append(0)  # 0 for normal
                    processed_count += 1
        
        print(f"Successfully processed {len(normal_features)} normal images")
        return normal_features, normal_labels
    
    def train_model(self, X, y):
        """Train the facial paralysis detection model"""
        print("Training the model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster training
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print("Fitting the model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Normal', 'Paralytic']))
        
        return model, X_test, y_test, y_pred
    
    def save_model(self, model):
        """Save the trained model"""
        print("Saving the model...")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save the model
        joblib.dump(model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Save model info
        model_info = {
            "model_type": "RandomForestClassifier",
            "image_size": self.image_size,
            "feature_count": model.n_features_in_,
            "n_estimators": model.n_estimators,
            "training_date": str(np.datetime64('now'))
        }
        
        with open("models/model_info_retrained.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print("Model training completed successfully!")
    
    def run_training(self):
        """Run the complete training process"""
        print("🚀 Starting Facial Paralysis Model Retraining...")
        print("=" * 60)
        
        # Load datasets
        paralytic_features, paralytic_labels = self.load_paralytic_images()
        normal_features, normal_labels = self.load_normal_images()
        
        if len(paralytic_features) == 0 or len(normal_features) == 0:
            print("❌ Error: Could not load sufficient data for training")
            return None
        
        # Combine all features and labels
        all_features = paralytic_features + normal_features
        all_labels = paralytic_labels + normal_labels
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"Total dataset size: {len(X)} images")
        print(f"Paralytic images: {sum(y)}")
        print(f"Normal images: {len(y) - sum(y)}")
        
        # Train model
        model, X_test, y_test, y_pred = self.train_model(X, y)
        
        # Save model
        self.save_model(model)
        
        print("=" * 60)
        print("✅ Model retraining completed successfully!")
        print(f"📊 Final accuracy: {model.score(X_test, y_test):.4f}")
        print(f"💾 Model saved to: {self.model_path}")
        
        return model

if __name__ == "__main__":
    trainer = SimpleFacialParalysisTrainer()
    model = trainer.run_training()
