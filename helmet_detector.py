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
        
        # Enhanced helmet colors in HSV with broader ranges
        self.helmet_colors = {
            'yellow': ([15, 50, 50], [35, 255, 255]),
            'red': ([0, 50, 50], [10, 255, 255]),
            'red2': ([170, 50, 50], [180, 255, 255]),  # Red wraps around HSV
            'blue': ([100, 50, 50], [130, 255, 255]),
            'white': ([0, 0, 150], [180, 30, 255]),
            'orange': ([10, 50, 50], [25, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'gray': ([0, 0, 50], [180, 30, 150])
        }
        
        self.confidence_threshold = 0.5
        self.helmet_threshold = 0.3  # Lowered threshold for better detection

    def _detect_helmet_in_region(self, image, bbox):
        """Detect helmet in a specific region using enhanced color and shape analysis"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract head region (upper portion of person) - increased region
        head_height = int((y2 - y1) * 0.4)  # Increased from 1/3 to 0.4
        head_region = image[y1:y1 + head_height, x1:x2]
        
        if head_region.size == 0:
            return False, 0.0
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Check for helmet colors with improved detection
        color_score = 0.0
        for color_name, (lower, upper) in self.helmet_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_ratio = np.sum(mask > 0) / mask.size
            color_score = max(color_score, color_ratio)
        
        # Enhanced shape detection
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        
        # Multiple shape detection methods
        shape_score = 0.0
        
        # Method 1: Circle detection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=25, minRadius=8, maxRadius=60  # Adjusted parameters
        )
        
        if circles is not None:
            shape_score = max(shape_score, min(1.0, len(circles[0]) * 0.4))
        
        # Method 2: Contour analysis for helmet-like shapes
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Check for circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Circularity threshold
                        shape_score = max(shape_score, circularity)
        
        # Method 3: Edge detection for helmet boundaries
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        shape_score = max(shape_score, edge_density * 0.5)
        
        # Combined score with adjusted weights (60% color, 40% shape)
        combined_score = 0.6 * color_score + 0.4 * shape_score
        
        # Additional logic: If we detect significant color but low shape, still consider it a helmet
        if color_score > 0.2 and combined_score > 0.15:
            combined_score = max(combined_score, 0.4)
        
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
                            'person_id': persons_detected + 1,
                            'bbox': bbox.tolist(),
                            'confidence': float(box.conf[0]),
                            'helmet_detected': int(has_helmet),  # Convert to int for JSON serialization
                            'helmet_confidence': float(confidence),
                            'status': 'Helmet' if has_helmet else 'No Helmet'
                        }
                        detections.append(detection)
                        
                        persons_detected += 1
                        if has_helmet:
                            helmets_detected += 1
        
        # Draw annotations on image
        annotated_image = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            color = (0, 255, 0) if detection['helmet_detected'] == 1 else (0, 0, 255)
            label = f"{detection['status']} ({detection['helmet_confidence']:.2f})"
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        filename = os.path.basename(image_path)
        result_path = os.path.join('results', f'result_{filename}')
        cv2.imwrite(result_path, annotated_image)
        
        # Update statistics
        self._update_stats(persons_detected, helmets_detected)
        
        # Create detailed summary
        compliance_rate = (helmets_detected / persons_detected * 100) if persons_detected > 0 else 0
        
        summary = {
            'total_persons': persons_detected,
            'helmet_count': helmets_detected,
            'no_helmet_count': persons_detected - helmets_detected,
            'compliance_rate': round(compliance_rate, 1),
            'text_summary': f"Detected {persons_detected} person(s). {helmets_detected} with helmets, {persons_detected - helmets_detected} without helmets. Compliance rate: {compliance_rate:.1f}%"
        }
        
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
                            label = f"{'Helmet' if has_helmet else 'No Helmet'} ({confidence:.2f})"
                            
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
        
        # Create detailed video summary
        compliance_rate = (total_helmets / total_persons * 100) if total_persons > 0 else 0
        
        summary = {
            'total_frames_analyzed': frame_count,
            'total_persons_detected': total_persons,
            'helmet_count': total_helmets,
            'no_helmet_count': total_persons - total_helmets,
            'average_compliance_rate': round(compliance_rate, 1),
            'text_summary': f"Analyzed {frame_count} frames. Detected {total_persons} person instances. {total_helmets} with helmets, {total_persons - total_helmets} without helmets. Overall compliance rate: {compliance_rate:.1f}%"
        }
        
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

    def get_statistics(self):
        """Get current system statistics"""
        total = self.stats['total_detections']
        helmets = self.stats['helmet_detections']
        compliance_rate = (helmets / total * 100) if total > 0 else 0
        
        return {
            'total_detections': total,
            'helmet_detections': helmets,
            'no_helmet_detections': self.stats['no_helmet_detections'],
            'compliance_rate': round(compliance_rate, 1)
        }