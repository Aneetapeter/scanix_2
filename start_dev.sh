#!/bin/bash

echo "Starting Scanix Development Environment"
echo "======================================"

echo ""
echo "Starting Python Flask Backend..."
cd backend
python run.py &
BACKEND_PID=$!

echo ""
echo "Waiting for backend to start..."
sleep 5

echo ""
echo "Starting Flutter Frontend..."
cd ..
flutter run -d web-server --web-port 3000 &
FLUTTER_PID=$!

echo ""
echo "Development environment started!"
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services..."

# Function to cleanup processes
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FLUTTER_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Wait for processes
wait
