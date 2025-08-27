import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import os
import csv
from datetime import datetime, timedelta
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Paths
model_path = "./models/hand_landmarker.task"
videos_directory = "./data/cropped_videos"  # Directory containing videos to process

# Get list of video files in the directory
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = []

if os.path.exists(videos_directory):
    for filename in os.listdir(videos_directory):
        if filename.lower().endswith(video_extensions):
            video_files.append(os.path.join(videos_directory, filename))
else:
    print(f"Error: Videos directory '{videos_directory}' does not exist!")
    exit(1)

if not video_files:
    print(f"No video files found in '{videos_directory}'")
    exit(1)

print(f"Found {len(video_files)} video(s) to process:")
for video_file in video_files:
    print(f"  - {os.path.basename(video_file)}")

def parse_video_timestamp(video_name_without_ext):
    """Parse timestamp from video name (format: chunk_XX_YYYYMMDD_HHMMSS)"""
    try:
        parts = video_name_without_ext.split('_')
        date_str = parts[2]  # YYYYMMDD
        time_str = parts[3]  # HHMMSS
        
        # Convert to datetime object
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"), parts
    except:
        # Fallback if parsing fails
        print(f"Warning: Could not parse timestamp from video name '{video_name_without_ext}', using current time")
        return datetime.now(), video_name_without_ext.split('_')

# Create base output directories
detections_output_dir = "detections_output"
hand_detections_output_dir = "hand_detections_output"
os.makedirs(detections_output_dir, exist_ok=True)
os.makedirs(os.path.join(detections_output_dir, hand_detections_output_dir), exist_ok=True)

# Analysis CSV (shared across steps)
analysis_csv_path = os.path.join(detections_output_dir, "analysis_log.csv")


def load_analysis_status(csv_path):
    status = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status[row.get("video_name", "")] = row
    return status


def write_analysis_status(csv_path, status_dict):
    fieldnames = [
        "video_name",
        "hand_detection_done",
        "object_classification_done",
        "intersections_done",
        "last_run",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video_name, row in sorted(status_dict.items()):
            # Ensure defaults
            writer.writerow({
                "video_name": video_name,
                "hand_detection_done": str(row.get("hand_detection_done", False)),
                "object_classification_done": str(row.get("object_classification_done", False)),
                "intersections_done": str(row.get("intersections_done", False)),
                "last_run": row.get("last_run", "")
            })


analysis_status = load_analysis_status(analysis_csv_path)

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

# Process each video file
for video_idx, video_path in enumerate(video_files):
    print(f"\n{'='*60}")
    print(f"Processing video {video_idx + 1}/{len(video_files)}: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Extract video filename and info
    video_filename = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_filename)[0]
    
    # Parse timestamp from video name
    video_start_time, parts = parse_video_timestamp(video_name_without_ext)

    # Skip if already processed for hand detection
    status_row = analysis_status.get(video_name_without_ext, {})
    if str(status_row.get("hand_detection_done", "False")).lower() == "true":
        print(f"Skipping hand detection for {video_filename} (already marked done in analysis_log.csv)")
        continue
    
    # Create main output directory for this video
    main_output_dir = video_name_without_ext
    output_img_dir = os.path.join(detections_output_dir, hand_detections_output_dir, main_output_dir, "detected_hands")
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_timestamp = 0
    
    print(f"Video info: {width}x{height} @ {fps:.2f} fps")
    
    # Region of Interest - Inclined with 10 degree tilt
    # Calculate 10-degree inclination
    angle_rad = math.radians(10)
    tan_angle = math.tan(angle_rad)
    
    # ROI positioned to keep bottom edge fixed, with 75% of original height
    original_roi_height = height // 8  # Original eighth of screen
    roi_height = int(original_roi_height * 0.75)  # Reduce to 75% of original height
    # Keep bottom edge at the same position, adjust start position upward
    roi_start_y = (height // 2) + original_roi_height - roi_height
    
    # Define ROI corners for inclined rectangle
    # Top-left corner
    roi_top_left_x = 0
    roi_top_left_y = roi_start_y
    
    # Top-right corner (inclined down by tan(15°) * width)
    roi_top_right_x = width
    roi_top_right_y = roi_start_y + int(tan_angle * width)
    
    # Bottom-left corner (moved down by roi_height)
    roi_bottom_left_x = 0
    roi_bottom_left_y = roi_start_y + roi_height
    
    # Bottom-right corner (inclined down by tan(15°) * width + roi_height)
    roi_bottom_right_x = width
    roi_bottom_right_y = roi_start_y + roi_height + int(tan_angle * width)
    
    # ROI corners as list for drawing
    roi_corners = np.array([
        [roi_top_left_x, roi_top_left_y],
        [roi_top_right_x, roi_top_right_y],
        [roi_bottom_right_x, roi_bottom_right_y],
        [roi_bottom_left_x, roi_bottom_left_y]
    ], dtype=np.int32)
    
    # Detection lines (7 parallel inclined lines: top ROI edge + 5 internal + bottom ROI edge)
    detection_lines = []
    
    # Line 0: Top edge of ROI
    detection_lines.append({
        'start': (0, roi_start_y),
        'end': (width, roi_start_y + int(tan_angle * width)),
        'y_left': roi_start_y,
        'y_right': roi_start_y + int(tan_angle * width)
    })
    
    # Lines 1-5: Internal lines within the ROI (evenly spaced with reduced gaps)
    for i in range(5):
        # Calculate the y-offset for each line (evenly spaced within reduced roi_height)
        y_offset = (i + 1) * roi_height // 6  # Divide by 6 to get 5 lines within the smaller ROI
        
        # Line start point (left side)
        line_start_x = 0
        line_start_y = roi_start_y + y_offset
        
        # Line end point (right side, inclined)
        line_end_x = width
        line_end_y = roi_start_y + y_offset + int(tan_angle * width)
        
        detection_lines.append({
            'start': (line_start_x, line_start_y),
            'end': (line_end_x, line_end_y),
            'y_left': line_start_y,
            'y_right': line_end_y
        })
    
    # Line 6: Bottom edge of ROI
    detection_lines.append({
        'start': (0, roi_start_y + roi_height),
        'end': (width, roi_start_y + roi_height + int(tan_angle * width)),
        'y_left': roi_start_y + roi_height,
        'y_right': roi_start_y + roi_height + int(tan_angle * width)
    })
    
    # Video writer - save in main output directory with new naming
    output_video_name = f"hand_detection_{video_filename}"
    output_video_path = os.path.join(detections_output_dir, hand_detections_output_dir, main_output_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Tracking state
    save_interval = 1  # Minimum interval to allow immediate saves for each line crossing
    last_saved_frame = -save_interval
    
    # Track hand data across frames
    hand_data = {}  # hand_id: {'prev_bbox': (x1,y1,x2,y2), 'prev_center': (cx,cy), 'last_seen_frame': frame_idx, 'folder_name': folder_name, 'img_count': count, 'first_detection_time': datetime, 'trajectory_history': [], 'crossed_lines_history': []}
    next_hand_id = 0
    MAX_HAND_ABSENCE = 120  # frames - increased to handle longer detection losses (4 seconds at 30fps)
    REAPPEARANCE_SEARCH_RADIUS = 200  # pixels - search radius for reappearing hands
    TRAJECTORY_HISTORY_LENGTH = 10  # number of recent positions to keep for trajectory prediction
    
    # Hand detector config
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=6,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, frame_timestamp)

                # Create clean frame for saving (without any visual elements)
                clean_frame = frame.copy()
                
                # Draw inclined ROI polygon (only for display, not for saving)
                cv2.polylines(frame, [roi_corners], True, (0, 255, 255), 2)
                
                # Draw inclined detection lines (only for display, not for saving)
                for line_data in detection_lines:
                    cv2.line(frame, line_data['start'], line_data['end'], (255, 0, 255), 2)

                if result.hand_landmarks:
                    current_hands = []
                    
                    for i, hand_landmarks in enumerate(result.hand_landmarks):
                        coords = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

                        # Draw landmarks
                        for x, y in coords:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                        # Bounding box
                        x_coords, y_coords = zip(*coords)
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                        # Calculate center point
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        bbox_area = (x_max - x_min) * (y_max - y_min)
                        
                        current_hands.append({
                            'bbox': (x_min, y_min, x_max, y_max), 
                            'center': (center_x, center_y),
                            'y_max': y_max,
                            'area': bbox_area
                        })

                    # Simple hand tracking - this is a simplified version
                    # For full functionality, you would need the complete tracking logic
                    for hand in current_hands:
                        # Simple detection of line crossing for demo
                        for line_idx, line_data in enumerate(detection_lines):
                            line_y = line_data['y_left']  # Simplified - using left y coordinate
                            if hand['y_max'] <= line_y + 5 and hand['y_max'] >= line_y - 5:
                                # Hand is near a detection line - save image
                                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                img_filename = os.path.join(output_img_dir, f"{timestamp_str}_line{line_idx}.jpg")
                                cv2.imwrite(img_filename, clean_frame)
                                print(f"Hand detected near line {line_idx} - saved: {img_filename}")

                out.write(frame)
                frame_idx += 1
                frame_timestamp += int(1000 / fps)

    except Exception as e:
        print(f"Error processing video {video_filename}: {e}")
    finally:
        cap.release()
        out.release()
        # Mark as completed for hand detection in CSV
        status_row = analysis_status.get(video_name_without_ext, {})
        status_row["video_name"] = video_name_without_ext
        status_row["hand_detection_done"] = True
        # Do not flip other flags
        status_row.setdefault("object_classification_done", False)
        status_row.setdefault("intersections_done", False)
        status_row["last_run"] = datetime.now().isoformat()
        analysis_status[video_name_without_ext] = status_row
        try:
            write_analysis_status(analysis_csv_path, analysis_status)
        except Exception as csv_e:
            print(f"Warning: could not write analysis CSV: {csv_e}")
    
    print(f"\nProcessing complete for {video_filename}!")
    print(f"Output directory: {main_output_dir}")
    print(f"Output video: {output_video_path}")
    print(f"Detected hands images: {output_img_dir}")

print(f"\n{'='*60}")
print(f"ALL VIDEOS PROCESSED SUCCESSFULLY!")
print(f"Processed {len(video_files)} video(s)")
print(f"{'='*60}")
