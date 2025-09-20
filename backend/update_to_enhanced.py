import os
import shutil

def update_to_enhanced_model():
    """Update backend to use the enhanced model"""
    print("🔄 Updating to Enhanced AI Model...")
    
    # Paths
    enhanced_model = 'models/enhanced_ai_model.pkl'
    enhanced_scaler = 'models/enhanced_scaler.pkl'
    current_model = 'models/ai_model.pkl'
    current_scaler = 'models/scaler.pkl'
    
    # Check if enhanced model exists
    if not os.path.exists(enhanced_model):
        print("❌ Enhanced model not found! Please run enhanced training first.")
        return False
    
    if not os.path.exists(enhanced_scaler):
        print("❌ Enhanced scaler not found! Please run enhanced training first.")
        return False
    
    try:
        # Backup current model
        if os.path.exists(current_model):
            shutil.copy2(current_model, 'models/ai_model_backup.pkl')
            print("✅ Backed up current model")
        
        if os.path.exists(current_scaler):
            shutil.copy2(current_scaler, 'models/scaler_backup.pkl')
            print("✅ Backed up current scaler")
        
        # Copy enhanced model to current
        shutil.copy2(enhanced_model, current_model)
        shutil.copy2(enhanced_scaler, current_scaler)
        
        print("✅ Enhanced model activated!")
        print("✅ Enhanced scaler activated!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating model: {e}")
        return False

if __name__ == '__main__':
    update_to_enhanced_model()
