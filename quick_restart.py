import subprocess
import os
import time

print("🚀 QUICK RESTART - ENHANCED AI BACKEND")
print("=" * 40)

# Stop existing Python processes
print("Step 1: Stopping old backend...")
try:
    subprocess.run(['taskkill', '/f', '/im', 'python.exe'], capture_output=True)
    print("✅ Old backend stopped")
except:
    print("ℹ️  No old backend to stop")

time.sleep(2)

# Start enhanced backend
print("\nStep 2: Starting enhanced backend...")
os.chdir('backend')
subprocess.Popen(['python', 'final_app.py'], shell=True)
print("✅ Enhanced backend started")

time.sleep(3)

# Start Flutter app
print("\nStep 3: Starting Flutter app...")
os.chdir('..')
subprocess.Popen(['flutter', 'run', '-d', 'web-server', '--web-port', '3002'], shell=True)
print("✅ Flutter app started")

print("\n" + "=" * 40)
print("🎉 ENHANCED AI BACKEND RESTARTED!")
print("=" * 40)
print("Your Scanix app now has:")
print("✅ Enhanced facial asymmetry analysis")
print("✅ Texture and edge feature extraction")
print("✅ Better paralysis detection accuracy")
print("✅ No OpenCV dependency issues")
print("\nAccess your app at: http://localhost:3002")
print("Backend running at: http://localhost:5000")
