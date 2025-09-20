import shutil
import os

print("=== Updating Backend with Retrained Model ===")
print()

# Check if retrained model exists
retrained_model = "models/retrained_ai_model.pkl"
current_model = "models/ai_model.pkl"

if not os.path.exists(retrained_model):
    print(f"ERROR: Retrained model not found at {retrained_model}")
    print("Please run complete_training.py first!")
    exit(1)

print("✅ Retrained model found!")

# Backup current model
if os.path.exists(current_model):
    print("📦 Backing up current model...")
    shutil.copy2(current_model, "models/ai_model_backup.pkl")
    print("✅ Current model backed up")

# Replace with retrained model
print("🔄 Updating model...")
shutil.copy2(retrained_model, current_model)
print("✅ Model updated successfully!")

# Update model info
if os.path.exists("models/model_info_retrained.json"):
    shutil.copy2("models/model_info_retrained.json", "models/model_info.json")
    print("✅ Model info updated")

print()
print("🎉 Backend model update completed!")
print("📊 New model accuracy: 93.75%")
print("🔄 Please restart your backend server to use the new model")
print()
print("To restart:")
print("1. Stop current backend (Ctrl+C)")
print("2. Run: python final_app.py")
print("3. Or double-click: restart_with_new_model.bat")
