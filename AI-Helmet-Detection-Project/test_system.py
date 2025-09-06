#!/usr/bin/env python3
"""
AI Safety Helmet Detection - System Test
Test all components of the helmet detection system
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile

def create_test_image():
    """Create a synthetic test image with people and helmets"""
    # Create a 640x480 image
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some people (simplified)
    # Person 1 with helmet
    draw.ellipse([100, 200, 150, 250], fill='pink')  # Head
    draw.ellipse([80, 180, 170, 200], fill='yellow')  # Helmet
    draw.rectangle([120, 250, 140, 350], fill='blue')  # Body
    
    # Person 2 without helmet
    draw.ellipse([300, 200, 350, 250], fill='pink')  # Head
    draw.rectangle([320, 250, 340, 350], fill='red')  # Body
    
    # Person 3 with helmet
    draw.ellipse([500, 200, 550, 250], fill='pink')  # Head
    draw.ellipse([480, 180, 570, 200], fill='orange')  # Helmet
    draw.rectangle([520, 250, 540, 350], fill='green')  # Body
    
    # Save test image
    test_image_path = 'test_image.jpg'
    img.save(test_image_path)
    print(f"✅ Test image created: {test_image_path}")
    return test_image_path

def test_helmet_detector():
    """Test the helmet detection functionality"""
    print("\n�� Testing Helmet Detector...")
    
    try:
        from helmet_detector import HelmetDetector
        
        # Create detector
        detector = HelmetDetector()
        print("✅ HelmetDetector initialized")
        
        # Create test image
        test_image = create_test_image()
        
        # Test detection
        result = detector.detect_image(test_image)
        
        if result['success']:
            print("✅ Image detection successful")
            print(f"�� Summary: {result['summary']}")
        else:
            print("❌ Image detection failed")
            
        # Cleanup
        if os.path.exists(test_image):
            os.remove(test_image)
            
    except Exception as e:
        print(f"❌ Helmet detector test failed: {e}")
        return False
    
    return True

def test_flask_app():
    """Test Flask application setup"""
    print("\n🌐 Testing Flask Application...")
    
    try:
        from app import app
        
        # Test basic configuration
        assert app.config['UPLOAD_FOLDER'] == 'uploads'
        assert app.config['RESULTS_FOLDER'] == 'results'
        print("✅ Flask app configuration correct")
        
        # Test routes exist
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200
            print("✅ Main route working")
            
            response = client.get('/api/stats')
            assert response.status_code == 200
            print("✅ Stats API working")
            
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        'flask', 'cv2', 'numpy', 'PIL', 'ultralytics',
        'torch', 'torchvision', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def cleanup_test_files():
    """Clean up any test files created"""
    test_files = ['test_image.jpg']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"�� Cleaned up: {file}")

def main():
    """Run all system tests"""
    print("🪖 AI Safety Helmet Detection - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Flask App", test_flask_app),
        ("Helmet Detector", test_helmet_detector)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n�� Running {test_name} Test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("�� Run 'python run.py' to start the application")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    cleanup_test_files()

if __name__ == "__main__":
    main()
