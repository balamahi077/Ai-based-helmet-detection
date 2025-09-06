#!/usr/bin/env python3
"""
AI Safety Helmet Detection - Startup Script
Automatically checks dependencies and starts the application
"""

import sys
import subprocess
import webbrowser
import time
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_dependencies():
    """Check and install required dependencies"""
    print("📦 Checking dependencies...")
    
    try:
        import flask
        import cv2
        import numpy
        import ultralytics
        print("✅ All dependencies are already installed")
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("📥 Installing dependencies...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"�� Created directory: {directory}")

def start_application():
    """Start the Flask application"""
    print("�� Starting AI Safety Helmet Detection...")
    print("�� Opening browser in 3 seconds...")
    
    # Wait a bit for the server to start
    time.sleep(3)
    
    try:
        webbrowser.open('http://localhost:5000')
        print("✅ Browser opened successfully")
    except:
        print("⚠️  Could not open browser automatically")
        print("🌐 Please open: http://localhost:5000")
    
    # Start the Flask app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    print("🪖 AI Safety Helmet Detection - Startup")
    print("=" * 50)
    
    check_python_version()
    check_dependencies()
    create_directories()
    start_application()
