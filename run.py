#!/usr/bin/env python3
"""
Startup script for AI Safety Helmet Detection System
This script provides an easy way to start the application with proper setup.
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'flask',
        'opencv-python',
        'torch',
        'ultralytics',
        'numpy',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'opencv-python':
                __import__('cv2')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run: pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are available")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f" Created directory: {directory}")

def start_application():
    """Start the Flask application"""
    print(" Starting AI Safety Helmet Detection System...")
    
    # Create directories
    create_directories()
    
    # Start the Flask app
    try:
        from flask_app import app
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        # Start browser thread
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("🌐 Starting Flask server...")
        print("📱 Opening browser automatically...")
        print("🔗 Manual access: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("Make sure flask_app.py exists and is properly configured")
        return False
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def main():
    """Main function"""
    print("🪖 AI Safety Helmet Detection System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Start application
    return start_application()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)