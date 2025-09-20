import shutil
import os

print("Updating backend to use retrained model...")

# Paths
retrained_model = "models/retrained_ai_model.pkl"
current_model = "models/ai_model.pkl"
backup_model = "models/ai_model_backup.pkl"

# Check if retrained model exists
if not os.path.exists(retrained_model):
    print(f"Retrained model not found at {retrained_model}")
    print("Please run complete_training.py first!")
    exit()

# Backup current model
if os.path.exists(current_model):
    print("Backing up current model...")
    shutil.copy2(current_model, backup_model)
    print(f"Current model backed up to {backup_model}")

# Replace with retrained model
print("Replacing model with retrained version...")
shutil.copy2(retrained_model, current_model)
print("Model updated successfully!")

# Update model info
retrained_info = "models/model_info_retrained.json"
current_info = "models/model_info.json"

if os.path.exists(retrained_info):
    shutil.copy2(retrained_info, current_info)
    print("Model info updated!")

print("Backend model update completed!")
print("Please restart the backend server to use the new model.")
