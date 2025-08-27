import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import os
from datetime import datetime, timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Paths
model_path = "./models/hand_landmarker.task"
videos_directory = "./data/cropped_videos"

# Setup MediaPipe (done once outside the loop)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

# Hand detector config (done once outside the loop)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=6,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

# Get all MP4 videos in the directory
video_files = [f for f in os.listdir(videos_directory) if f.lower().endswith('.mp4')]
print(f"Found {len(video_files)} videos to process: {video_files}")

# Process each video
for video_filename in video_files:
    video_path = os.path.join(videos_directory, video_filename)
    print(f"\n{'='*60}")
    print(f"Processing video: {video_filename}")
    print(f"{'='*60}")
    
    # Extract video filename and info
    video_name_without_ext = os.path.splitext(video_filename)[0]

    # Parse timestamp from video name (format: chunk_XX_YYYYMMDD_HHMMSS)
    try:
        parts = video_name_without_ext.split('_')
        date_str = parts[2]  # YYYYMMDD
        time_str = parts[3]  # HHMMSS
        
        # Convert to datetime object
        video_start_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except:
        # Fallback if parsing fails
        video_start_time = datetime.now()
        print(f"Warning: Could not parse timestamp from video name, using current time")

    # Create main output directory named after video
    main_output_dir = video_name_without_ext
    detections_output_dir = "detections_output"
    hand_detections_output_dir = "hand_detections_output"
    os.makedirs(detections_output_dir, exist_ok=True)
    os.makedirs(os.path.join(detections_output_dir, hand_detections_output_dir), exist_ok=True)

    # Create subdirectory for detected hands
    output_img_dir = os.path.join(detections_output_dir, hand_detections_output_dir, main_output_dir, "detected_hands")
    os.makedirs(output_img_dir, exist_ok=True)

    # Video setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_timestamp = 0

    # Region of Interest - Inclined with 10 degree tilt
    import math

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

    # Start hand detection processing for this video
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

            # Clean up old/inactive hands first (but be more conservative for hands with crossing history)
            hands_to_remove = []
            for hand_id, data in hand_data.items():
                frames_absent = frame_idx - data.get('last_seen_frame', 0)
                crossed_lines_history = data.get('crossed_lines_history', [])
                
                # Extend MAX_HAND_ABSENCE for hands with good crossing history
                if len(crossed_lines_history) >= 2:
                    # Hands that have crossed multiple lines get more time to reappear
                    extended_absence_limit = MAX_HAND_ABSENCE * 2  # Double the time
                else:
                    extended_absence_limit = MAX_HAND_ABSENCE
                
                if frames_absent > extended_absence_limit:
                    hands_to_remove.append(hand_id)
            
            for hand_id in hands_to_remove:
                crossed_lines_count = len(hand_data[hand_id].get('crossed_lines_history', []))
                print(f"Retiring inactive hand {hand_id} (absent for {frame_idx - hand_data[hand_id].get('last_seen_frame', 0)} frames, crossed {crossed_lines_count} lines)")
                del hand_data[hand_id]
            
            # Advanced hand matching with trajectory prediction and reappearance detection
            def predict_hand_position(tracked_hand_data, frames_absent):
                """Predict where a hand might appear based on trajectory history"""
                trajectory = tracked_hand_data.get('trajectory_history', [])
                if len(trajectory) < 2:
                    return tracked_hand_data.get('prev_center')
                
                # Calculate average velocity from recent trajectory
                velocities = []
                for i in range(1, min(len(trajectory), 5)):  # Use last 5 positions
                    dx = trajectory[i][0] - trajectory[i-1][0]
                    dy = trajectory[i][1] - trajectory[i-1][1]
                    velocities.append((dx, dy))
                
                if not velocities:
                    return tracked_hand_data.get('prev_center')
                
                # Average velocity
                avg_vx = sum(v[0] for v in velocities) / len(velocities)
                avg_vy = sum(v[1] for v in velocities) / len(velocities)
                
                # Predict position based on trajectory
                last_center = tracked_hand_data.get('prev_center')
                if last_center:
                    predicted_x = last_center[0] + avg_vx * frames_absent
                    predicted_y = last_center[1] + avg_vy * frames_absent
                    return (predicted_x, predicted_y)
                
                return None
            
            def calculate_hand_similarity(current_hand, tracked_hand_data, frame_idx):
                """Calculate similarity score between current and tracked hand with trajectory prediction"""
                if 'prev_center' not in tracked_hand_data or 'prev_bbox' not in tracked_hand_data:
                    return float('inf')
                
                curr_center = current_hand['center']
                curr_bbox = current_hand['bbox']
                curr_area = current_hand['area']
                
                # Calculate frames absent
                frames_absent = frame_idx - tracked_hand_data.get('last_seen_frame', frame_idx)
                
                # Use predicted position if hand has been absent
                if frames_absent > 1:
                    predicted_center = predict_hand_position(tracked_hand_data, frames_absent)
                    if predicted_center:
                        # Use predicted position for comparison
                        prev_center = predicted_center
                    else:
                        prev_center = tracked_hand_data['prev_center']
                else:
                    prev_center = tracked_hand_data['prev_center']
                
                prev_bbox = tracked_hand_data['prev_bbox']
                prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
                
                # Distance between centers (use predicted position if available)
                center_dist = ((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)**0.5
                
                # Size similarity (area ratio)
                size_ratio = max(curr_area, prev_area) / max(min(curr_area, prev_area), 1)
                
                # Bounding box overlap (IoU) - use original bbox for IoU calculation
                x1 = max(curr_bbox[0], prev_bbox[0])
                y1 = max(curr_bbox[1], prev_bbox[1])
                x2 = min(curr_bbox[2], prev_bbox[2])
                y2 = min(curr_bbox[3], prev_bbox[3])
                
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                union = curr_area + prev_area - intersection
                iou = intersection / max(union, 1)
                
                # Trajectory consistency bonus
                trajectory_bonus = 0
                if len(tracked_hand_data.get('trajectory_history', [])) >= 2:
                    # Check if current position follows expected trajectory
                    expected_pos = predict_hand_position(tracked_hand_data, frames_absent)
                    if expected_pos:
                        expected_dist = ((curr_center[0] - expected_pos[0])**2 + (curr_center[1] - expected_pos[1])**2)**0.5
                        if expected_dist < 50:  # Within 50 pixels of expected position
                            trajectory_bonus = 30  # Bonus for trajectory consistency
                
                # Line crossing history bonus - prefer hands that crossed similar lines
                line_history_bonus = 0
                crossed_lines_history = tracked_hand_data.get('crossed_lines_history', [])
                if crossed_lines_history:
                    # Bonus for hands that have crossed lines (suggesting continuous motion)
                    line_history_bonus = min(len(crossed_lines_history) * 10, 50)  # Up to 50 point bonus
                
                # Adjust scoring based on absence duration
                absence_penalty = frames_absent * 2  # Penalty increases with absence duration
                
                # Combined score (lower is better)
                # More lenient scoring for reappearing hands
                base_score = center_dist + (size_ratio - 1) * 30 - iou * 80  # Reduced penalties
                final_score = base_score + absence_penalty - trajectory_bonus - line_history_bonus
                
                return final_score
            
            # Create cost matrix for hand matching
            current_hands_matched = [False] * len(current_hands)
            
            # Phase 1: Try to match with recently seen hands (standard matching)
            for current_idx, current_hand in enumerate(current_hands):
                if current_hands_matched[current_idx]:
                    continue
                    
                best_match_id = None
                best_score = float('inf')
                
                # Find best matching tracked hand among recently seen hands
                for hand_id, data in hand_data.items():
                    # Skip hands that were already matched this frame
                    if data.get('matched_this_frame', False):
                        continue
                    
                    # Check if hand was seen recently (within reasonable time)
                    frames_absent = frame_idx - data.get('last_seen_frame', frame_idx)
                    if frames_absent > MAX_HAND_ABSENCE:
                        continue
                        
                    score = calculate_hand_similarity(current_hand, data, frame_idx)
                    
                    # Dynamic threshold based on absence duration
                    # Recently seen hands: stricter threshold
                    # Longer absent hands: more lenient threshold
                    if frames_absent <= 5:
                        threshold = 80  # Strict for recently seen hands
                    elif frames_absent <= 30:
                        threshold = 120  # Moderate for short absence
                    else:
                        threshold = 200  # Lenient for longer absence
                    
                    if score < threshold and score < best_score:
                        best_score = score
                        best_match_id = hand_id
                
                # Phase 2: If no match found, try more aggressive reappearance detection
                if best_match_id is None:
                    # Look for hands that might have reappeared after longer absence
                    for hand_id, data in hand_data.items():
                        if data.get('matched_this_frame', False):
                            continue
                        
                        frames_absent = frame_idx - data.get('last_seen_frame', frame_idx)
                        if frames_absent <= MAX_HAND_ABSENCE:
                            continue  # Already checked in phase 1
                        
                        # Check if this could be a reappearance based on trajectory and line crossing history
                        crossed_lines_history = data.get('crossed_lines_history', [])
                        if len(crossed_lines_history) > 0:  # Only consider hands that have crossed lines before
                            score = calculate_hand_similarity(current_hand, data, frame_idx)
                            
                            # Additional bonus for sequential line crossing pattern
                            # If hand was crossing lines in sequence, give extra consideration
                            pattern_bonus = 0
                            if len(crossed_lines_history) >= 2:
                                # Check if lines were crossed in sequence (descending order: 6,5,4,3,2,1,0)
                                recent_lines = [entry['line_idx'] for entry in crossed_lines_history[-3:]]  # Last 3 crossings
                                if all(recent_lines[i] > recent_lines[i+1] for i in range(len(recent_lines)-1)):
                                    pattern_bonus = 100  # Strong bonus for sequential pattern
                                    print(f"Hand {hand_id} shows sequential line crossing pattern: {recent_lines}")
                            
                            # Very lenient threshold for reappearance detection
                            # Focus on hands that have crossing history and sequential patterns
                            adjusted_score = score - pattern_bonus
                            if adjusted_score < 400 and adjusted_score < best_score:
                                best_score = adjusted_score
                                best_match_id = hand_id
                                print(f"Potential reappearance: Hand {hand_id} (absent for {frames_absent} frames, score: {score:.1f}, pattern bonus: {pattern_bonus})")
                
                # Phase 3: Create new hand if no match found
                if best_match_id is None:
                    best_match_id = next_hand_id
                    next_hand_id += 1
                    
                    hand_data[best_match_id] = {
                        'img_count': 0,
                        'last_saved_frame': -save_interval,
                        'folder_created': False,
                        'last_seen_frame': frame_idx,
                        'trajectory_history': [],
                        'crossed_lines_history': []
                    }
                    print(f"New hand detected (ID: {best_match_id}) - tracking started")
                else:
                    # Mark this tracked hand as matched
                    hand_data[best_match_id]['matched_this_frame'] = True
                    frames_absent = frame_idx - hand_data[best_match_id].get('last_seen_frame', frame_idx)
                    if frames_absent > 1:
                        print(f"Hand {best_match_id} reappeared after {frames_absent} frames (score: {best_score:.1f})")
                
                current_hands_matched[current_idx] = True
                
                # Get previous position for line crossing detection
                prev_y_max = hand_data[best_match_id].get('prev_y_max', None)
                current_y_max = current_hand['y_max']
                
                # Check if hand crossed any inclined detection line from below to above
                crossed_lines = []
                if prev_y_max is not None:
                    prev_center_x = hand_data[best_match_id].get('prev_center', (0, 0))[0]
                    current_center_x = current_hand['center'][0]
                    
                    for line_idx, line_data in enumerate(detection_lines):
                        # Calculate y-coordinate of the inclined line at the hand's x-position
                        # Linear interpolation between line start and end points
                        line_start_x, line_start_y = line_data['start']
                        line_end_x, line_end_y = line_data['end']
                        
                        # Calculate line y-coordinate at previous hand center x-position
                        if line_end_x != line_start_x:
                            prev_line_y = line_start_y + (line_end_y - line_start_y) * (prev_center_x - line_start_x) / (line_end_x - line_start_x)
                        else:
                            prev_line_y = line_start_y
                        
                        # Calculate line y-coordinate at current hand center x-position
                        if line_end_x != line_start_x:
                            current_line_y = line_start_y + (line_end_y - line_start_y) * (current_center_x - line_start_x) / (line_end_x - line_start_x)
                        else:
                            current_line_y = line_start_y
                        
                        # Check if hand crossed this inclined line from below to above
                        if prev_y_max > prev_line_y and current_y_max <= current_line_y:
                            crossed_lines.append({
                                'line_idx': line_idx,
                                'line_y': current_line_y,
                                'line_data': line_data
                            })
                
                                # Save image for each line that was crossed
                for crossed_line_info in crossed_lines:
                    # Check if this specific line was already crossed recently (to avoid duplicate saves)
                    line_already_crossed = False
                    if 'crossed_lines' in hand_data[best_match_id]:
                        for prev_crossed_line in hand_data[best_match_id]['crossed_lines']:
                            if (prev_crossed_line['line_idx'] == crossed_line_info['line_idx'] and 
                                frame_idx - prev_crossed_line['frame'] < save_interval):
                                line_already_crossed = True
                                break
                    
                    if not line_already_crossed:
                        # Create folder only when first line crossing occurs
                        if not hand_data[best_match_id]['folder_created']:
                            # Calculate timestamp for first line crossing
                            seconds_elapsed = frame_idx / fps
                            first_crossing_time = video_start_time + timedelta(seconds=seconds_elapsed)
                            timestamp_str = first_crossing_time.strftime("%Y%m%d_%H%M%S")
                            
                            # Create folder name for this hand
                            folder_name = f"chunk_{parts[1]}_{timestamp_str}"
                            hand_folder_path = os.path.join(output_img_dir, folder_name)
                            os.makedirs(hand_folder_path, exist_ok=True)
                            
                            # Update hand data with folder info
                            hand_data[best_match_id]['folder_name'] = folder_name
                            hand_data[best_match_id]['folder_path'] = hand_folder_path
                            hand_data[best_match_id]['folder_created'] = True
                            hand_data[best_match_id]['first_crossing_time'] = first_crossing_time
                            
                            print(f"Hand {best_match_id} crossed line for first time - created folder: {folder_name}")
                        
                        # Calculate timestamp for this frame with milliseconds to avoid overwrites
                        seconds_elapsed = frame_idx / fps
                        frame_time = video_start_time + timedelta(seconds=seconds_elapsed)
                        timestamp_str = frame_time.strftime("%Y%m%d_%H%M%S")
                        milliseconds = int((seconds_elapsed % 1) * 1000)
                        
                        # Create unique filename with milliseconds and hand ID to prevent overwrites
                        img_filename = os.path.join(hand_data[best_match_id]['folder_path'], 
                                                  f"chunk_{parts[1]}_{timestamp_str}_{milliseconds:03d}_hand{best_match_id}.jpg")
                        
                        # Save clean image (without bounding boxes or lines)
                        cv2.imwrite(img_filename, clean_frame)
                        hand_data[best_match_id]['img_count'] += 1
                        hand_data[best_match_id]['last_saved_frame'] = frame_idx
                        
                        # Record hand coordinates for this image
                        hand_coords = {
                            'image_name': os.path.basename(img_filename),
                            'hand_id': best_match_id,
                            'bbox': current_hand['bbox'],  # (x_min, y_min, x_max, y_max)
                            'center': current_hand['center'],  # (center_x, center_y)
                            'y_max': current_hand['y_max'],
                            'crossed_line_y': crossed_line_info['line_y'],
                            'crossed_line_idx': crossed_line_info['line_idx'],
                            'frame_idx': frame_idx,
                            'timestamp': frame_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        }
                        
                        # Save hand coordinates to file
                        coords_file = os.path.join(hand_data[best_match_id]['folder_path'], 'hand_coordinates.txt')
                        
                        # Write header if file doesn't exist
                        if not os.path.exists(coords_file):
                            with open(coords_file, 'w', encoding='utf-8') as f:
                                f.write("image_name,hand_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,center_x,center_y,y_max,crossed_line_y,crossed_line_idx,frame_idx,timestamp\n")
                        
                        # Append coordinates
                        with open(coords_file, 'a', encoding='utf-8') as f:
                            f.write(f"{hand_coords['image_name']},{hand_coords['hand_id']},{hand_coords['bbox'][0]},{hand_coords['bbox'][1]},{hand_coords['bbox'][2]},{hand_coords['bbox'][3]},{hand_coords['center'][0]},{hand_coords['center'][1]},{hand_coords['y_max']},{hand_coords['crossed_line_y']},{hand_coords['crossed_line_idx']},{hand_coords['frame_idx']},{hand_coords['timestamp']}\n")
                        
                        # Record this line crossing to avoid duplicates
                        if 'crossed_lines' not in hand_data[best_match_id]:
                            hand_data[best_match_id]['crossed_lines'] = []
                        hand_data[best_match_id]['crossed_lines'].append({
                            'line_idx': crossed_line_info['line_idx'],
                            'line_y': crossed_line_info['line_y'],
                            'frame': frame_idx
                        })
                        
                        # Add to crossed lines history for trajectory analysis
                        if 'crossed_lines_history' not in hand_data[best_match_id]:
                            hand_data[best_match_id]['crossed_lines_history'] = []
                        hand_data[best_match_id]['crossed_lines_history'].append({
                            'line_idx': crossed_line_info['line_idx'],
                            'line_y': crossed_line_info['line_y'],
                            'frame': frame_idx,
                            'hand_center': current_hand['center']
                        })
                        
                        print(f"Hand {best_match_id} crossed inclined line {crossed_line_info['line_idx']} at y={crossed_line_info['line_y']:.1f} - saved clean image: {img_filename} (score: {best_score:.1f})")
                
                # Update trajectory history
                if 'trajectory_history' not in hand_data[best_match_id]:
                    hand_data[best_match_id]['trajectory_history'] = []
                
                trajectory_history = hand_data[best_match_id]['trajectory_history']
                trajectory_history.append(current_hand['center'])
                
                # Keep only recent trajectory history
                if len(trajectory_history) > TRAJECTORY_HISTORY_LENGTH:
                    trajectory_history.pop(0)
                
                # Update hand tracking data
                hand_data[best_match_id].update({
                    'prev_y_max': current_y_max,
                    'prev_center': current_hand['center'],
                    'prev_bbox': current_hand['bbox'],
                    'last_seen_frame': frame_idx,
                    'trajectory_history': trajectory_history
                })
            
            # Reset match flags for next frame
            for hand_id in hand_data:
                hand_data[hand_id]['matched_this_frame'] = False

            out.write(frame)
            frame_idx += 1
            frame_timestamp += int(1000 / fps)

    cap.release()
    out.release()

    print(f"\nProcessing complete for {video_filename}!")
    print(f"Output directory: {main_output_dir}")
    print(f"Output video: {output_video_path}")
    print(f"Detected hands images: {output_img_dir}")
    print(f"Total hands detected: {len(hand_data)}")

    total_images = 0
    hands_with_folders = 0
    for hand_id, data in hand_data.items():
        if data.get('folder_created', False):
            print(f"  Hand {hand_id}: {data['img_count']} images in folder '{data['folder_name']}'")
            total_images += data['img_count']
            hands_with_folders += 1
        else:
            print(f"  Hand {hand_id}: tracked but never crossed lines (no folder created)")

    print(f"Hands that crossed lines: {hands_with_folders}")
    print(f"Total images saved: {total_images}")

print("\n" + "="*60)
print("ALL VIDEOS PROCESSING COMPLETE!")
print("="*60)
