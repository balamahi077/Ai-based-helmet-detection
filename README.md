# AI Safety Helmet Detection System

A comprehensive Python-based computer vision system for detecting safety helmets in images, videos, and real-time camera feeds. This project uses advanced AI models to ensure workplace safety compliance.

## 🎯 Features

### Core Functionality
- **Image Detection**: Upload and analyze images for helmet detection
- **Video Analysis**: Process video files for comprehensive safety analysis
- **Real-time Detection**: Live camera feed analysis with instant alerts
- **Multi-format Support**: JPG, PNG, JPEG, MP4, AVI, MOV files
- **Advanced AI Models**: YOLO-based person detection with custom helmet analysis

### Dashboard & Analytics
- **Real-time Statistics**: Live compliance rates and detection counts
- **Interactive Dashboard**: Modern, responsive web interface
- **Detection Logs**: Detailed logs with timestamps and confidence scores
- **Export Capabilities**: Download detection results and statistics
- **Visual Annotations**: Bounding boxes and labels on detected objects

### Safety Features
- **Instant Alerts**: Real-time notifications for safety violations
- **Compliance Tracking**: Monitor helmet usage over time
- **Risk Assessment**: Identify high-risk areas and time periods
- **Report Generation**: Automated safety compliance reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ACT-ACADEMY
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the dashboard**
   - Open your browser and go to `http://localhost:5000`
   - The system will automatically download required AI models on first run

## 📁 Project Structure

```
ACT-ACADEMY/
├── app.py                 # Main Flask application
├── helmet_detector.py     # Core detection logic
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── index.html        # Main dashboard
│   └── realtime.html     # Real-time detection page
├── uploads/              # Uploaded files (auto-created)
├── results/              # Detection results (auto-created)
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
FLASK_ENV=development
FLASK_DEBUG=True
UPLOAD_FOLDER=uploads
RESULTS_FOLDER=results
MAX_FILE_SIZE=16777216
```

### Model Configuration
The system uses:
- **YOLOv8**: For person detection
- **Custom Helmet Detection**: Color and shape analysis
- **OpenCV**: For image processing and video analysis

## 📖 Usage Guide

### 1. Image Detection
1. Navigate to the main dashboard
2. Click on "Image Detection" tab
3. Upload an image file (JPG, PNG, JPEG)
4. Click "Detect Helmets"
5. View results with bounding boxes and statistics

### 2. Video Analysis
1. Select "Video Analysis" tab
2. Upload a video file (MP4, AVI, MOV)
3. Click "Analyze Video"
4. Monitor progress and view comprehensive results

### 3. Real-time Detection
1. Click "Real-time Detection" tab
2. Select your camera from the dropdown
3. Click "Start Detection"
4. Monitor live feed with real-time alerts

### 4. Dashboard Analytics
- **Statistics Cards**: View total detections, compliance rates
- **Detection Log**: Real-time log of all detections
- **Export Data**: Download statistics and logs

## 🤖 AI Model Details

### Person Detection
- **Model**: YOLOv8 (Ultralytics)
- **Classes**: Person detection (class 0)
- **Confidence**: Configurable threshold (default: 0.5)

### Helmet Detection Algorithm
1. **Person Region Extraction**: Crop detected person areas
2. **Color Analysis**: HSV color space analysis for common helmet colors
3. **Shape Detection**: Circular/oval shape detection using Hough Circles
4. **Combined Scoring**: Weighted combination of color and shape scores
5. **Threshold Classification**: Final helmet/no-helmet decision

### Supported Helmet Colors
- White, Yellow, Red, Blue, Green, Orange
- Configurable HSV ranges for each color
- Adaptive thresholding for lighting conditions

## 📊 API Endpoints

### Detection Endpoints
- `POST /detect` - Image helmet detection
- `POST /detect_video` - Video helmet analysis
- `GET /video_feed` - Real-time video stream

### Data Endpoints
- `GET /api/stats` - Get detection statistics
- `GET /uploads/<filename>` - Serve uploaded files
- `GET /results/<filename>` - Serve result files

### Web Pages
- `GET /` - Main dashboard
- `GET /realtime` - Real-time detection page

## 🔍 Detection Algorithm

### Step-by-Step Process
1. **Input Processing**: Load and preprocess image/video
2. **Person Detection**: Use YOLO to detect people
3. **Region Analysis**: Extract person bounding boxes
4. **Helmet Detection**:
   - Convert to HSV color space
   - Apply color masks for helmet colors
   - Detect circular shapes
   - Calculate combined confidence score
5. **Result Generation**: Annotate image and generate statistics

### Confidence Scoring
```
Combined Score = (Color Score × 0.7) + (Shape Score × 0.3)
Helmet Detected = Combined Score > Threshold (default: 0.1)
```

## 🛠️ Customization

### Adding New Helmet Colors
Edit `helmet_colors` in `helmet_detector.py`:

```python
self.helmet_colors = {
    'custom_color': ([H_min, S_min, V_min], [H_max, S_max, V_max]),
    # ... existing colors
}
```

### Adjusting Detection Sensitivity
Modify thresholds in `_detect_helmet_in_region()`:

```python
# Color weight vs shape weight
combined_score = (max_color_score * 0.7) + (circle_score * 0.3)

# Detection threshold
helmet_detected = combined_score > 0.1  # Adjust this value
```

### Model Selection
Change YOLO model in `helmet_detector.py`:

```python
# For faster processing (less accurate)
self.person_model = YOLO('yolov8n.pt')

# For better accuracy (slower)
self.person_model = YOLO('yolov8s.pt')
```

## 📈 Performance Optimization

### For Production Use
1. **GPU Acceleration**: Install CUDA for faster processing
2. **Model Optimization**: Use TensorRT or ONNX for inference
3. **Batch Processing**: Process multiple images simultaneously
4. **Caching**: Cache model weights and results
5. **Load Balancing**: Deploy multiple instances for high traffic

### Memory Management
- Process videos in chunks
- Clear GPU memory after processing
- Use efficient data structures
- Implement garbage collection

## 🐛 Troubleshooting

### Common Issues

**1. Model Download Fails**
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**2. CUDA/GPU Issues**
```bash
# Check CUDA installation
nvidia-smi
# Install CPU-only version if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Memory Issues**
- Reduce batch size
- Process smaller images
- Use lighter YOLO model (yolov8n instead of yolov8l)

**4. Camera Access Issues**
- Check camera permissions
- Try different camera index (0, 1, 2)
- Update camera drivers

### Debug Mode
Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

### File Upload Security
- File type validation
- Size limits (16MB default)
- Secure file storage
- Virus scanning (recommended)

### Data Privacy
- Local processing (no data sent to external servers)
- Secure file deletion
- Access control for sensitive areas
- GDPR compliance for EU users

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🔮 Future Enhancements

### Planned Features
- **Multi-class Detection**: Different helmet types (hard hat, bike helmet, etc.)
- **Face Recognition**: Individual identification for compliance tracking
- **Mobile App**: iOS/Android companion app
- **Cloud Integration**: AWS/Azure deployment options
- **Advanced Analytics**: Machine learning insights and predictions
- **Integration APIs**: Connect with existing safety management systems

### Research Areas
- **Improved Accuracy**: Better helmet detection algorithms
- **Real-time Optimization**: Faster processing for live feeds
- **Edge Computing**: Deploy on IoT devices
- **Federated Learning**: Privacy-preserving model training

