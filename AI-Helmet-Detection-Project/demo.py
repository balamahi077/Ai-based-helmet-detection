#!/usr/bin/env python3
"""
AI Safety Helmet Detection - Demo Script
Demonstrate the system using available images
"""

import os
import glob
from helmet_detector import HelmetDetector

def list_available_images():
    """List all available images in the current directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(glob.glob(ext))
    
    return images

def demonstrate_system():
    """Demonstrate the helmet detection system"""
    print("🪖 AI Safety Helmet Detection - Demo")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = HelmetDetector()
        print("✅ Helmet detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Find available images
    images = list_available_images()
    
    if not images:
        print("⚠️  No images found in current directory")
        print("📁 Please add some images (.jpg, .png, etc.) to test the system")
        return
    
    print(f"📸 Found {len(images)} image(s): {', '.join(images)}")
    print("\n�� Starting detection demo...\n")
    
    total_persons = 0
    total_helmets = 0
    total_no_helmets = 0
    
    # Process each image
    for i, image_path in enumerate(images, 1):
        print(f"��️  Processing image {i}/{len(images)}: {image_path}")
        
        try:
            # Detect helmets
            result = detector.detect_image(image_path)
            
            if result['success']:
                summary = result['summary']
                total_persons += summary['persons_detected']
                total_helmets += summary['helmets_detected']
                total_no_helmets += summary['no_helmets']
                
                print(f"   ✅ Persons: {summary['persons_detected']}")
                print(f"   ✅ Helmets: {summary['helmets_detected']}")
                print(f"   ✅ No Helmets: {summary['no_helmets']}")
                print(f"   📊 Compliance: {summary['compliance_rate']}%")
                print(f"   💾 Result saved: {result['result_filename']}")
            else:
                print(f"   ❌ Detection failed")
                
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
        
        print()
    
    # Overall statistics
    print("📊 Overall Demo Statistics:")
    print("=" * 30)
    print(f"📸 Images processed: {len(images)}")
    print(f"👥 Total persons detected: {total_persons}")
    print(f"🪖 Total helmets found: {total_helmets}")
    print(f"⚠️  Total no helmets: {total_no_helmets}")
    
    if total_persons > 0:
        overall_compliance = (total_helmets / total_persons) * 100
        print(f"�� Overall compliance rate: {overall_compliance:.1f}%")
    
    print("\n🎯 Recommendations:")
    if total_no_helmets > 0:
        print(f"⚠️  {total_no_helmets} safety violations detected")
        print("🔧 Consider implementing stricter safety protocols")
    else:
        print("✅ Excellent safety compliance!")
    
    print("\n🚀 To start the web interface, run: python run.py")

def show_system_info():
    """Show system information"""
    print("\nℹ️  System Information:")
    print("=" * 30)
    print("🪖 AI Safety Helmet Detection System")
    print("�� Built with: Python, Flask, OpenCV, YOLOv8")
    print("🌐 Web Interface: HTML/CSS/JavaScript")
    print("📁 Project Structure:")
    print("   ├── app.py (Flask server)")
    print("   ├── helmet_detector.py (AI engine)")
    print("   ├── templates/ (Web interface)")
    print("   ├── uploads/ (Uploaded files)")
    print("   └── results/ (Detection results)")

if __name__ == "__main__":
    show_system_info()
    demonstrate_system()
