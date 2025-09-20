#!/bin/bash
echo "Starting Scanix AI System..."
echo

echo "Starting Backend Server..."
cd backend
python app.py &
BACKEND_PID=$!

echo "Waiting for backend to start..."
sleep 3

echo "Starting Flutter App..."
cd ..
flutter run &
FLUTTER_PID=$!

echo
echo "Scanix AI System is running..."
echo "Backend: http://localhost:5000"
echo "Frontend: Will open automatically"
echo
echo "Press Ctrl+C to stop both services"

# Wait for user interrupt
trap "kill $BACKEND_PID $FLUTTER_PID; exit" INT
wait
