import os
import numpy as np
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tarfile
import shutil
import json
import cv2
from sklearn.preprocessing import StandardScaler

class EnhancedFacialParalysisTrainer:
    def __init__(self):
        self.paralytic_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'archive (4)', 'Strokefaces', 'droopy')
        self.normal_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'normal face')
        self.lfw_tgz_path = os.path.join(self.normal_path, 'lfw-funneled.tgz')
        self.lfw_extracted_path = os.path.join(self.normal_path, 'lfw_funneled')
        self.model_save_path = 'models/enhanced_ai_model.pkl'
        self.scaler_save_path = 'models/enhanced_scaler.pkl'
        self.model_info_path = 'models/enhanced_model_info.json'
        self.image_size = (64, 64)  # Higher resolution for better features
        self.max_images_per_class = 500  # More data for better training

    def _extract_lfw(self):
        """Extract LFW dataset if not already extracted"""
        if os.path.exists(self.lfw_extracted_path):
            print("✅ LFW dataset already extracted")
            return
        
        if not os.path.exists(self.lfw_tgz_path):
            print("❌ LFW dataset not found!")
            return
        
        print("📦 Extracting LFW dataset...")
        try:
            with tarfile.open(self.lfw_tgz_path, 'r:gz') as tar:
                tar.extractall(self.normal_path)
            print("✅ LFW dataset extracted successfully")
        except Exception as e:
            print(f"❌ Error extracting LFW: {e}")

    def _detect_face(self, image):
        """Detect and crop face from image"""
        try:
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(img_cv, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_cv.shape[1] - x, w + 2 * padding)
                h = min(img_cv.shape[0] - y, h + 2 * padding)
                
                # Crop face
                face = img_cv[y:y+h, x:x+w]
                return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            else:
                return image  # Return original if no face detected
        except:
            return image  # Return original if face detection fails

    def _extract_facial_features(self, image):
        """Extract enhanced facial features for paralysis detection"""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Resize to target size
            img_resized = cv2.resize(img_array, self.image_size)
            
            # Extract asymmetry features (key for paralysis detection)
            left_half = img_resized[:, :self.image_size[1]//2]
            right_half = img_resized[:, self.image_size[1]//2:]
            
            # Flip right half to compare with left
            right_flipped = cv2.flip(right_half, 1)
            
            # Calculate asymmetry metrics
            asymmetry = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float)))
            
            # Extract edge features (facial contours)
            edges = cv2.Canny(img_resized, 50, 150)
            edge_density = np.sum(edges > 0) / (self.image_size[0] * self.image_size[1])
            
            # Extract texture features
            texture_features = []
            for kernel_size in [3, 5, 7]:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                filtered = cv2.filter2D(img_resized, -1, kernel)
                texture_features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.var(filtered)
                ])
            
            # Combine all features
            basic_features = img_resized.flatten()  # Basic pixel features
            asymmetry_features = [asymmetry, edge_density]
            combined_features = np.concatenate([basic_features, asymmetry_features, texture_features])
            
            return combined_features
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Fallback to basic features
            return np.array(image.convert('L').resize(self.image_size)).flatten()

    def _load_images_from_folder(self, folder, label, limit=None):
        """Load and preprocess images from folder"""
        images = []
        labels = []
        
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            return images, labels
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if limit:
            files = files[:limit]
        
        print(f"📁 Loading {len(files)} images from {os.path.basename(folder)}...")
        
        for i, filename in enumerate(files):
            try:
                img_path = os.path.join(folder, filename)
                image = Image.open(img_path)
                
                # Detect and crop face
                face_image = self._detect_face(image)
                
                # Extract enhanced features
                features = self._extract_facial_features(face_image)
                
                images.append(features)
                labels.append(label)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(files)} images")
                    
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
        
        print(f"✅ Loaded {len(images)} images successfully")
        return images, labels

    def _load_lfw_images(self, limit=None):
        """Load images from LFW dataset"""
        images = []
        labels = []
        
        if not os.path.exists(self.lfw_extracted_path):
            print("❌ LFW dataset not found!")
            return images, labels
        
        person_dirs = [d for d in os.listdir(self.lfw_extracted_path) 
                      if os.path.isdir(os.path.join(self.lfw_extracted_path, d))]
        
        print(f"📁 Found {len(person_dirs)} person directories in LFW")
        
        total_loaded = 0
        for person_dir in person_dirs:
            if limit and total_loaded >= limit:
                break
                
            person_path = os.path.join(self.lfw_extracted_path, person_dir)
            person_files = [f for f in os.listdir(person_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images per person to ensure diversity
            max_per_person = min(3, len(person_files))
            person_files = person_files[:max_per_person]
            
            for filename in person_files:
                if limit and total_loaded >= limit:
                    break
                    
                try:
                    img_path = os.path.join(person_path, filename)
                    image = Image.open(img_path)
                    
                    # Detect and crop face
                    face_image = self._detect_face(image)
                    
                    # Extract enhanced features
                    features = self._extract_facial_features(face_image)
                    
                    images.append(features)
                    labels.append(0)  # Normal face
                    total_loaded += 1
                    
                    if total_loaded % 50 == 0:
                        print(f"  Processed {total_loaded} LFW images")
                        
                except Exception as e:
                    print(f"  Error processing {person_dir}/{filename}: {e}")
                    continue
        
        print(f"✅ Loaded {len(images)} LFW images successfully")
        return images, labels

    def run_enhanced_training(self):
        """Run enhanced training with better feature extraction"""
        print("🚀 ENHANCED FACIAL PARALYSIS MODEL TRAINING")
        print("=" * 60)
        
        # Extract LFW dataset
        self._extract_lfw()
        
        # Load paralytic images
        print("\n📊 Loading paralytic images...")
        paralytic_images, paralytic_labels = self._load_images_from_folder(
            self.paralytic_path, 1, self.max_images_per_class
        )
        
        # Load normal images from LFW
        print("\n📊 Loading normal images from LFW...")
        normal_images, normal_labels = self._load_lfw_images(self.max_images_per_class)
        
        if len(paralytic_images) == 0 or len(normal_images) == 0:
            print("❌ Not enough data for training!")
            return None
        
        # Combine datasets
        X = np.array(paralytic_images + normal_images)
        y = np.array(paralytic_labels + normal_labels)
        
        print(f"\n📈 Dataset Summary:")
        print(f"  Total images: {len(X)}")
        print(f"  Paralytic: {np.sum(y == 1)}")
        print(f"  Normal: {np.sum(y == 0)}")
        print(f"  Feature dimension: {X.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🔄 Training split:")
        print(f"  Training: {len(X_train)} images")
        print(f"  Testing: {len(X_test)} images")
        
        # Scale features
        print("\n⚖️ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train enhanced model
        print("\n🤖 Training enhanced Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=20,      # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 RESULTS:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Paralytic']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  Normal predicted as Normal: {cm[0,0]}")
        print(f"  Normal predicted as Paralytic: {cm[0,1]}")
        print(f"  Paralytic predicted as Normal: {cm[1,0]}")
        print(f"  Paralytic predicted as Paralytic: {cm[1,1]}")
        
        # Save model and scaler
        print(f"\n💾 Saving enhanced model...")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(model, self.model_save_path)
        joblib.dump(scaler, self.scaler_save_path)
        
        # Save model info
        model_info = {
            "accuracy": float(accuracy),
            "classification_report": classification_report(y_test, y_pred, target_names=['Normal', 'Paralytic'], output_dict=True),
            "confusion_matrix": cm.tolist(),
            "feature_dimension": int(X.shape[1]),
            "dataset_size": int(len(X)),
            "paralytic_images": int(np.sum(y == 1)),
            "normal_images": int(np.sum(y == 0)),
            "model_type": "Enhanced RandomForestClassifier",
            "image_size": self.image_size,
            "features_used": ["pixel_values", "asymmetry", "edge_density", "texture"]
        }
        
        with open(self.model_info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"✅ Enhanced model saved to {self.model_save_path}")
        print(f"✅ Scaler saved to {self.scaler_save_path}")
        print(f"✅ Model info saved to {self.model_info_path}")
        
        print(f"\n🎉 ENHANCED TRAINING COMPLETED!")
        print(f"   Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return model

if __name__ == '__main__':
    trainer = EnhancedFacialParalysisTrainer()
    trainer.run_enhanced_training()
