# 🪖 AI Safety Helmet Detection

A comprehensive AI-powered system for detecting safety helmets in images, videos, and real-time camera feeds. Built with Python, Flask, and YOLOv8 for workplace safety compliance.

## 🚀 Features

- **Image Detection**: Upload and analyze images for helmet compliance
- **Video Analysis**: Process video files with frame-by-frame detection
- **Real-time Detection**: Live camera feed with instant helmet detection
- **Modern Dashboard**: Beautiful web interface built with HTML/CSS/JavaScript
- **Analytics**: Comprehensive statistics and compliance reporting
- **Export Results**: Save annotated images and videos with detection overlays

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Flask-CORS
- **AI/ML**: YOLOv8 (Ultralytics), OpenCV, PyTorch
- **Computer Vision**: OpenCV, NumPy, PIL
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing**: NumPy, Matplotlib, Seaborn

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Quick Start

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd AI-Helmet-Detection-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python run.py
```

The application will automatically:
- Check Python version compatibility
- Install missing dependencies
- Create necessary directories
- Start the Flask server
- Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
AI-Helmet-Detection-Project/
├── app.py                 # Main Flask application
├── helmet_detector.py     # AI detection engine
├── run.py                # Startup script
├── demo.py               # Demonstration script
├── test_system.py        # System testing
├── generate_ppt.py       # PPT generation
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── templates/           # HTML templates
│   ├── index.html       # Main dashboard
│   └── realtime.html    # Real-time detection page
├── uploads/             # Uploaded files (auto-created)
├── results/             # Detection results (auto-created)
└── yolov8n.pt          # YOLOv8 model file
```

## 🎯 Usage Guide

### Web Interface

1. **Image Detection**:
   - Go to the "Image Detection" tab
   - Upload an image file (JPG, PNG, etc.)
   - Click "Detect Helmets"
   - View results with bounding boxes and statistics

2. **Video Analysis**:
   - Go to the "Video Analysis" tab
   - Upload a video file (MP4, AVI, etc.)
   - Click "Analyze Video"
   - Wait for processing and view results

3. **Real-time Detection**:
   - Go to the "Real-time" tab
   - Click "Open Real-time Page"
   - Select camera and start detection
   - Monitor live statistics and alerts

### Command Line

```bash
# Test the system
python test_system.py

# Run demo with available images
python demo.py

# Generate presentation
python generate_ppt.py
```

## 🎯 AI Model Details

### Detection Algorithm

1. **Person Detection**: Uses YOLOv8n model to detect people in images/videos
2. **Helmet Detection**: Custom algorithm combining:
   - **Color Analysis**: HSV color space detection for common helmet colors
   - **Shape Detection**: Hough Circle transform for helmet-like shapes
   - **Combined Score**: 70% color + 30% shape confidence

### Supported Helmet Colors

- Yellow
- Red
- Blue
- White
- Orange

### Configuration

Edit `helmet_detector.py` to adjust:
- Confidence thresholds
- Color ranges
- Detection parameters

## 🎯 API Endpoints

### Main Routes

- `GET /` - Main dashboard
- `GET /realtime` - Real-time detection page
- `POST /detect` - Image detection endpoint
- `POST /detect_video` - Video analysis endpoint
- `GET /api/stats` - System statistics
- `GET /uploads/<filename>` - Serve uploaded files
- `GET /results/<filename>` - Serve result files

### Example API Usage

```python
import requests

# Detect helmets in image
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/detect', files=files)
    result = response.json()
    print(result['summary'])

# Get system statistics
response = requests.get('http://localhost:5000/api/stats')
stats = response.json()
print(stats)
```

## 🎯 Customization

### Adding New Helmet Colors

Edit the `helmet_colors` dictionary in `helmet_detector.py`:

```python
self.helmet_colors = {
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'red': ([0, 100, 100], [10, 255, 255]),
    'blue': ([100, 100, 100], [130, 255, 255]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'orange': ([10, 100, 100], [20, 255, 255]),
    'green': ([40, 100, 100], [80, 255, 255]),  # Add new color
}
```

### Adjusting Detection Sensitivity

Modify thresholds in `helmet_detector.py`:

```python
self.confidence_threshold = 0.5    # Person 
