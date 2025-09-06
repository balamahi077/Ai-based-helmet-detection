#!/usr/bin/env python3
"""
AI Safety Helmet Detection Engine
Core detection logic using YOLOv8 and custom helmet detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import json

class HelmetDetector:
    def __init__(self):
        """Initialize the helmet detection system"""
        self.person_model = YOLO('yolov8n.pt')  # Load YOLOv8 for person detection
        self.stats = {
            'total_detections': 0,
            'helmet_detections': 0,
            'no_helmet_detections': 0,
        }
        
        # Define helmet colors in HSV
        self.helmet_colors = {
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'orange': ([10, 100, 100], [20, 255, 255])
        }
        
        self.confidence_threshold = 0.5
        self.helmet_threshold = 0.6

    def _detect_helmet_in_region(self, image, bbox):
        """Detect helmet in a specific region using color and shape analysis"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract head region (upper portion of person)
        head_region = image[y1:y1 + (y2-y1)//3, x1:x2]
        if head_region.size == 0:
            return False, 0.0
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Check for helmet colors
        color_score = 0.0
        for color_name, (lower, upper) in self.helmet_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_ratio = np.sum(mask > 0) / mask.size
            color_score = max(color_score, color_ratio)
        
        # Check for circular/helmet-like shapes
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        shape_score = 0.0
        if circles is not None:
            shape_score = min(1.0, len(circles[0]) * 0.3)
        
        # Combined score (70% color, 30% shape)
        combined_score = 0.7 * color_score + 0.3 * shape_score
        
        return combined_score > self.helmet_threshold, combined_score

    def detect_image(self, image_path):
        """Detect helmets in a single image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect persons using YOLOv8
        results = self.person_model(image, classes=[0])  # class 0 is person
        
        detections = []
        persons_detected = 0
        helmets_detected = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if box.conf[0] > self.confidence_threshold:
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Detect helmet in this person region
                        has_helmet, confidence = self._detect_helmet_in_region(image, bbox)
                        
                        detection = {
                            'bbox': bbox.tolist(),
                            'confidence': float(box.conf[0]),
                            'has_helmet': int(has_helmet),  # Convert to int for JSON serialization
                            'helmet_confidence': float(confidence)
                        }
                        detections.append(detection)
                        
                        persons_detected += 1
                        if has_helmet:
                            helmets_detected += 1
        
        # Draw annotations on image
        annotated_image = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            color = (0, 255, 0) if detection['has_helmet'] == 1 else (0, 0, 255)
            label = f"Helmet: {detection['helmet_confidence']:.2f}" if detection['has_helmet'] == 1 else "No Helmet"
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        filename = os.path.basename(image_path)
        result_path = os.path.join('results', f'result_{filename}')
        cv2.imwrite(result_path, annotated_image)
        
        # Update statistics
        self._update_stats(persons_detected, helmets_detected)
        
        # Create summary
        summary = self._create_summary(persons_detected, helmets_detected)
        
        return {
            'success': True,
            'detections': detections,
            'summary': summary,
            'result_filename': f'result_{filename}'
        }

    def detect_video(self, video_path):
        """Detect helmets in a video file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output video
        filename = os.path.basename(video_path)
        output_path = os.path.join('results', f'result_{filename}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_persons = 0
        total_helmets = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons in frame
            results = self.person_model(frame, classes=[0])
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.conf[0] > self.confidence_threshold:
                            bbox = box.xyxy[0].cpu().numpy()
                            has_helmet, confidence = self._detect_helmet_in_region(frame, bbox)
                            
                            # Draw annotations
                            x1, y1, x2, y2 = map(int, bbox)
                            color = (0, 255, 0) if has_helmet else (0, 0, 255)
                            label = f"Helmet: {confidence:.2f}" if has_helmet else "No Helmet"
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            total_persons += 1
                            if has_helmet:
                                total_helmets += 1
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Update statistics
        self._update_stats(total_persons, total_helmets)
        
        # Create video summary
        summary = self._create_video_summary(total_persons, total_helmets, frame_count)
        
        return {
            'success': True,
            'summary': summary,
            'result_filename': f'result_{filename}'
        }

    def _update_stats(self, persons_detected, helmets_detected):
        """Update system statistics"""
        self.stats['total_detections'] += persons_detected
        self.stats['helmet_detections'] += helmets_detected
        self.stats['no_helmet_detections'] += (persons_detected - helmets_detected)

    def _create_summary(self, persons_detected, helmets_detected):
        """Create a summary of detection results"""
        if persons_detected == 0:
            return "No persons detected in the image."
        
        compliance_rate = (helmets_detected / persons_detected) * 100
        return f"Detected {persons_detected} person(s). {helmets_detected} with helmets, {persons_detected - helmets_detected} without helmets. Compliance rate: {compliance_rate:.1f}%"

    def _create_video_summary(self, total_persons, total_helmets, frame_count):
        """Create a summary of video analysis results"""
        if total_persons == 0:
            return f"Analyzed {frame_count} frames. No persons detected in the video."
        
        compliance_rate = (total_helmets / total_persons) * 100
        return f"Analyzed {frame_count} frames. Detected {total_persons} person instances. {total_helmets} with helmets, {total_persons - total_helmets} without helmets. Overall compliance rate: {compliance_rate:.1f}%"

    def get_statistics(self):
        """Get current system statistics"""
        return self.stats#!/usr/bin/env python3
"""
AI Safety Helmet Detection - Flask Application
Main web server for the helmet detection system
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from helmet_detector import HelmetDetector
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize helmet detector
detector = HelmetDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    """Real-time detection page"""
    return render_template('realtime.html')

@app.route('/detect', methods=['POST'])
def detect_image():
    """Detect helmets in uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = detector.detect_image(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detect_video', methods=['POST'])
def detect_video():
    """Detect helmets in uploaded video"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = detector.detect_video(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/video_feed')
def video_feed():
    """Real-time video feed endpoint"""
    # This would integrate with actual camera feed
    # For now, return a placeholder
    return jsonify({'status': 'video_feed_placeholder'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    stats = detector.get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)