To run the code:
- cd storeyes_v2
- storeyes_env/Scripts/activate



# StoreYes v2 - Hand Tracking & Object Detection System

A comprehensive computer vision system for real-time hand tracking, object detection, and interaction analysis in retail/service environments.

## Features

### 🤚 Advanced Hand Tracking
- **Improved tracking persistence** - Handles temporary detection losses
- **Trajectory prediction** - Predicts hand positions during brief disappearances  
- **Sequential line crossing detection** - Tracks hands moving through defined regions
- **Multi-hand support** - Tracks up to 6 hands simultaneously
- **ROI-based detection** - Configurable region of interest with inclined detection lines

### 🎯 Object Classification
- **YOLO-based object detection** - High-accuracy object recognition
- **Hand-object intersection analysis** - Determines which objects hands are interacting with
- **Padded bounding box detection** - Improved intersection detection with configurable padding
- **Confidence-based filtering** - Adjustable detection thresholds

### 📊 Analysis & Reporting
- **Frequency-based counting** - Handles duplicate frames intelligently
- **Multi-level summaries** - Global, video-level, and hand-level analysis
- **JSON exports** - Machine-readable analysis results
- **Intersection statistics** - Detailed IoU and confidence metrics
- **Visual annotations** - Annotated output images with bounding boxes

## Quick Start

### Prerequisites
```bash

pip install mediapipe opencv-python ultralytics numpy
```

### Usage

1. **Hand Detection & Tracking**
```bash
python hand_detection.py
```
- Processes video files and tracks hand movements
- Saves detected hand images when crossing detection lines
- Outputs tracking data and annotated video

2. **Object Classification**
```bash
python object_classification.py
```
- Runs YOLO object detection on hand images
- Analyzes hand-object intersections
- Generates annotated images and CSV reports

3. **Intersection Analysis**
```bash
python analyze_intersections.py
```
- Comprehensive analysis of detection results
- Creates video-level and hand-level summaries
- Exports JSON data for further processing

## Project Structure

```
storeyes_v2/
├── hand_detection.py           # Main hand tracking script
├── object_classification.py    # YOLO object detection
├── analyze_intersections.py    # Results analysis
├── hand_landmarker.task       # MediaPipe hand detection model
├── detections_output/         # Generated results
│   ├── hand_detections_output/
│   └── objects_classification_output/
└── test_scripts/              # Testing utilities
```

## Configuration

### Hand Detection Parameters
- `MAX_HAND_ABSENCE`: Frames to keep tracking lost hands (default: 120)
- `roi_height`: Height of detection region (75% of original)
- Detection lines: 7 parallel inclined lines for crossing detection

### Object Classification Parameters  
- `CONF_THRESHOLD`: Minimum detection confidence (default: 0.25)
- `HAND_BBOX_PADDING`: Padding for intersection detection (default: 25px)
- `PADDED_IOU_THRESHOLD`: IoU threshold for intersections (default: 0.04)

### Analysis Parameters
- `IOU_THRESHOLD`: Minimum IoU for significant intersections (default: 0.05)

## Key Improvements

### Enhanced Hand Tracking
- **Increased absence tolerance**: From 60 to 120 frames
- **Trajectory prediction**: Uses movement history to predict positions
- **Multi-phase matching**: Progressive thresholds for reappearance detection
- **Line crossing patterns**: Bonus scoring for sequential movements

### Better Object Detection
- **Padded bounding boxes**: Improved intersection detection
- **Dynamic thresholds**: Adaptive scoring based on absence duration
- **Frequency-based counting**: Eliminates duplicate frame issues

### Comprehensive Analysis
- **Video-level summaries**: Organized by video chunks
- **JSON exports**: Structured data for integration
- **Multiple output formats**: Both human and machine readable

## Output Files

### Hand Detection Output
- `detected_hands/<hand_folder>/`: Images of detected hands
- `hand_coordinates.txt`: Detailed tracking data
- `hand_detection_<video>.mp4`: Annotated video

### Object Classification Output
- `object_detections.csv`: Raw detection data
- `*_annotated.jpg`: Images with bounding boxes
- `<hand_folder>_summary.json`: Structured analysis results

### Analysis Output
- `global_summary.txt`: Video-level summaries
- `*_summary.json`: Detailed hand-level analysis
- `*_intersecting_detections.csv`: Filtered results
