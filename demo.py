#!/usr/bin/env python3
"""
Demonstration script for AI Safety Helmet Detection System
This script demonstrates the system using existing images in the workspace.
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def list_available_images():
    """List all available images in the workspace"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(file)
    
    return images

def demonstrate_system():
    """Demonstrate the helmet detection system"""
    print("🎯 AI Safety Helmet Detection System - Demonstration")
    print("=" * 60)
    
    # List available images
    images = list_available_images()
    
    if not images:
        print("❌ No images found in the workspace")
        print("Please add some images (JPG, PNG, JPEG) to test the system")
        return False
    
    print(f"📸 Found {len(images)} image(s) in workspace:")
    for i, image in enumerate(images, 1):
        print(f"   {i}. {image}")
    
    print("\n🚀 Initializing Helmet Detection System...")
    
    try:
        # Import and initialize the helmet detector
        from helmet_detector import HelmetDetector
        detector = HelmetDetector()
        print("✅ Helmet detector initialized successfully")
        
        # Process each image
        total_detections = 0
        total_helmets = 0
        
        for i, image_file in enumerate(images, 1):
            print(f"\n📷 Processing image {i}/{len(images)}: {image_file}")
            print("-" * 40)
            
            try:
                # Detect helmets in the image
                result = detector.detect_image(image_file)
                
                # Display results
                summary = result['summary']
                print(f"   👥 Persons detected: {summary['total_persons']}")
                print(f"   🪖 Helmets detected: {summary['helmet_count']}")
                print(f"   ⚠️  No helmets: {summary['no_helmet_count']}")
                print(f"   📊 Compliance rate: {summary['compliance_rate']:.1f}%")
                
                # Save annotated result
                result_filename = f"demo_result_{i}_{image_file}"
                cv2.imwrite(result_filename, result['annotated_image'])
                print(f"   💾 Result saved as: {result_filename}")
                
                # Update totals
                total_detections += summary['total_persons']
                total_helmets += summary['helmet_count']
                
            except Exception as e:
                print(f"   ❌ Error processing {image_file}: {str(e)}")
        
        # Display overall statistics
        print("\n" + "=" * 60)
        print("📈 OVERALL DEMONSTRATION RESULTS")
        print("=" * 60)
        
        overall_stats = detector.get_statistics()
        print(f"🎯 Total detections across all images: {overall_stats['total_detections']}")
        print(f"🪖 Total helmets detected: {overall_stats['helmet_detections']}")
        print(f"⚠️  Total no-helmet violations: {overall_stats['no_helmet_detections']}")
        print(f"📊 Overall compliance rate: {overall_stats['compliance_rate']:.1f}%")
        
        # Safety assessment
        if overall_stats['compliance_rate'] >= 90:
            safety_level = "🟢 EXCELLENT"
        elif overall_stats['compliance_rate'] >= 75:
            safety_level = "🟡 GOOD"
        elif overall_stats['compliance_rate'] >= 50:
            safety_level = "🟠 MODERATE"
        else:
            safety_level = "🔴 POOR"
        
        print(f"🛡️  Safety Level: {safety_level}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if overall_stats['compliance_rate'] < 90:
            print("   • Implement mandatory helmet policies")
            print("   • Conduct safety training sessions")
            print("   • Install helmet detection cameras")
            print("   • Regular safety audits")
        else:
            print("   • Maintain current safety standards")
            print("   • Continue monitoring compliance")
            print("   • Regular safety refresher training")
        
        print("\n🎉 Demonstration completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run 'python run.py' to start the web interface")
        print("   2. Open http://localhost:5000 in your browser")
        print("   3. Upload your own images for testing")
        print("   4. Try the real-time detection feature")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check if you have sufficient disk space for model downloads")
        print("   3. Ensure you have a working internet connection for model downloads")
        return False

def show_system_info():
    """Show system information and capabilities"""
    print("\n🔧 SYSTEM INFORMATION")
    print("-" * 30)
    print("🎯 Purpose: AI-powered safety helmet detection")
    print("🤖 AI Models: YOLOv8 + Custom helmet detection")
    print("📱 Interface: Web-based dashboard")
    print("📊 Features: Image, video, and real-time detection")
    print("🛡️  Safety: Real-time alerts and compliance tracking")
    print("📈 Analytics: Detailed statistics and reporting")

def main():
    """Main demonstration function"""
    print("🎯 AI Safety Helmet Detection System")
    print("Advanced computer vision for workplace safety")
    print("=" * 60)
    
    # Show system info
    show_system_info()
    
    # Run demonstration
    success = demonstrate_system()
    
    if success:
        print("\n✅ Demonstration completed successfully!")
        print("🚀 Ready to use the full system!")
    else:
        print("\n❌ Demonstration encountered issues")
        print("🔧 Please check the troubleshooting guide in README.md")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)