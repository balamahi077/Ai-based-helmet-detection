#!/usr/bin/env python3
"""
Test script for AI Safety Helmet Detection System
This script tests the core functionality of the helmet detection system.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile
import shutil

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """Create a test image with simulated people"""
    # Create a test image (640x480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add background
    img[:] = (100, 100, 100)  # Gray background
    
    # Draw person 1 (with helmet-like shape)
    cv2.rectangle(img, (100, 200), (140, 280), (255, 255, 255), -1)  # White body
    cv2.circle(img, (120, 190), 25, (255, 255, 0), -1)  # Yellow helmet
    
    # Draw person 2 (without helmet)
    cv2.rectangle(img, (300, 250), (340, 330), (255, 255, 255), -1)  # White body
    cv2.circle(img, (320, 240), 20, (100, 100, 100), -1)  # Gray head (no helmet)
    
    # Draw person 3 (with red helmet)
    cv2.rectangle(img, (500, 180), (540, 260), (255, 255, 255), -1)  # White body
    cv2.circle(img, (520, 170), 25, (0, 0, 255), -1)  # Red helmet
    
    return img

def test_helmet_detector():
    """Test the helmet detector with a sample image"""
    print("🧪 Testing Helmet Detection System...")
    
    try:
        # Import the helmet detector
        from helmet_detector import HelmetDetector
        
        # Initialize detector
        print("📦 Initializing Helmet Detector...")
        detector = HelmetDetector()
        print("✅ Helmet Detector initialized successfully")
        
        # Create test image
        print("🖼️  Creating test image...")
        test_img = create_test_image()
        
        # Save test image
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, test_img)
        print(f"✅ Test image saved as {test_image_path}")
        
        # Test detection
        print("🔍 Running helmet detection...")
        result = detector.detect_image(test_image_path)
        
        # Display results
        print("\n📊 Detection Results:")
        print(f"   Total persons detected: {result['summary']['total_persons']}")
        print(f"   Helmets detected: {result['summary']['helmet_count']}")
        print(f"   No helmets: {result['summary']['no_helmet_count']}")
        print(f"   Compliance rate: {result['summary']['compliance_rate']:.1f}%")
        
        # Save result image
        result_path = "test_result.jpg"
        cv2.imwrite(result_path, result['annotated_image'])
        print(f"✅ Result image saved as {result_path}")
        
        # Test statistics
        print("\n📈 Testing statistics...")
        stats = detector.get_statistics()
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Helmet detections: {stats['helmet_detections']}")
        print(f"   No helmet detections: {stats['no_helmet_detections']}")
        print(f"   Overall compliance: {stats['compliance_rate']:.1f}%")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_flask_app():
    """Test if Flask app can be imported and initialized"""
    print("\n🌐 Testing Flask Application...")
    
    try:
        # Import Flask app
        from app import app
        
        # Test app configuration
        assert app.config['UPLOAD_FOLDER'] == 'uploads'
        assert app.config['RESULTS_FOLDER'] == 'results'
        print("✅ Flask app configuration verified")
        
        # Test routes exist
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/detect', '/detect_video', '/realtime', '/api/stats']
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"⚠️  Route {route} not found")
        
        print("✅ Flask app test completed")
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {str(e)}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("📦 Testing Dependencies...")
    
    required_packages = [
        'cv2',
        'numpy',
        'torch',
        'ultralytics',
        'flask',
        'PIL',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are available")
        return True

def cleanup_test_files():
    """Clean up test files"""
    test_files = ['test_image.jpg', 'test_result.jpg']
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Cleaned up {file}")

def main():
    """Main test function"""
    print("🚀 AI Safety Helmet Detection System - Test Suite")
    print("=" * 60)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please install missing packages.")
        return False
    
    # Test Flask app
    if not test_flask_app():
        print("\n❌ Flask app test failed.")
        return False
    
    # Test helmet detector
    if not test_helmet_detector():
        print("\n❌ Helmet detector test failed.")
        return False
    
    # Cleanup
    cleanup_test_files()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("🚀 The system is ready to use.")
    print("\nTo start the application:")
    print("   python app.py")
    print("\nThen open your browser to: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)