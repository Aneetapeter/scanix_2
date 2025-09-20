#!/bin/bash

echo "Starting Scanix AI Application..."

# Start Backend
echo "Starting backend server..."
(cd backend && python improved_main.py) &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait a moment for the backend to initialize
echo "Waiting for backend to initialize (5 seconds)..."
sleep 5

# Start Frontend
echo "Starting Flutter frontend..."
flutter run &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Scanix AI Application started. Press Ctrl+C to stop both."

# Function to kill background processes on exit
cleanup() {
  echo "Stopping backend (PID: $BACKEND_PID) and frontend (PID: $FRONTEND_PID)..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  echo "Processes stopped."
}

# Trap Ctrl+C and call cleanup function
trap cleanup SIGINT

wait $BACKEND_PID
wait $FRONTEND_PID