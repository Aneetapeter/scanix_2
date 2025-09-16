#!/usr/bin/env python3
"""
Startup script for Scanix backend server
"""

import os
import sys
from app import app, load_model

def main():
    """Main function to start the Flask server"""
    print("Starting Scanix Backend Server...")
    print("=" * 50)
    
    # Load the AI model
    print("Loading AI model...")
    load_model()
    
    if not app.config.get('model_loaded', False):
        print("Warning: Model failed to load. Some features may not work.")
    else:
        print("✓ Model loaded successfully")
    
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

if __name__ == '__main__':
    main()
