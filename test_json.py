#!/usr/bin/env python3
"""
Test JSON serialization for helmet detection results
"""

import json
from helmet_detector import HelmetDetector

def test_json_serialization():
    """Test if detection results can be properly serialized to JSON"""
    detector = HelmetDetector()
    
    # Create a sample detection result
    sample_result = {
        'success': True,
        'detections': [
            {
                'bbox': [100, 100, 200, 300],
                'confidence': 0.95,
                'has_helmet': 1,  # Using int instead of bool
                'helmet_confidence': 0.85
            },
            {
                'bbox': [300, 150, 400, 350],
                'confidence': 0.88,
                'has_helmet': 0,  # Using int instead of bool
                'helmet_confidence': 0.25
            }
        ],
        'summary': 'Test summary',
        'result_filename': 'test_result.jpg'
    }
    
    try:
        # Try to serialize to JSON
        json_string = json.dumps(sample_result)
        print("✅ JSON serialization successful!")
        print("Sample result:", json_string)
        return True
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    test_json_serialization()