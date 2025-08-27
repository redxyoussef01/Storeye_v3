import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

import cv2
from ultralytics import YOLO

# Shared analysis CSV to avoid re-processing
ANALYSIS_CSV: str = "detections_output/analysis_log.csv"


def load_analysis_status(csv_path: Path) -> Dict[str, Dict]:
    status: Dict[str, Dict] = {}
    if csv_path.exists():
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status[row.get("video_name", "")] = row
    return status


def write_analysis_status(csv_path: Path, status_dict: Dict[str, Dict]) -> None:
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
            writer.writerow({
                "video_name": video_name,
                "hand_detection_done": str(row.get("hand_detection_done", False)),
                "object_classification_done": str(row.get("object_classification_done", False)),
                "intersections_done": str(row.get("intersections_done", False)),
                "last_run": row.get("last_run", "")
            })

# ---- Configuration (edit these variables) ----
# Path to YOLO model file (e.g., best.pt)
MODEL: str = "./models/best_n11.pt"

# Input directory (hand detection outputs)
HAND_DETECTIONS_INPUT: str = "detections_output/hand_detections_output"

# Output directory for object classification
OBJECTS_CLASSIFICATION_OUTPUT: str = "detections_output/objects_classification_output"

# Confidence threshold for filtering detections
CONF_THRESHOLD: float = 0.25

# Hand bounding box padding (in pixels) for better intersection detection
HAND_BBOX_PADDING: int = 25

# Lower IoU threshold for padded hand intersections (more lenient)
PADDED_IOU_THRESHOLD: float = 0.04

# Also save cropped detections per image
SAVE_CROPS: bool = False

# Draw padded hand bbox on annotated images for debugging
DRAW_PADDED_HAND_BBOX: bool = True

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Generate deterministic color for class ID (BGR for OpenCV)"""
    rng = (class_id * 37 + 17) % 255
    return (int(50 + (rng * 3) % 205), int(80 + (rng * 5) % 175), int(100 + (rng * 7) % 155))


def draw_detections(image, boxes, class_ids, confidences, class_names):
    """Draw bounding boxes and labels on image"""
    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, class_ids, confidences):
        color = get_color_for_class(int(cls_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[int(cls_id)]}:{conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, max(0, y1 - label_h - 8)), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )


def save_crops_for_image(image, boxes, class_ids, confidences, class_names, crops_dir: Path, image_stem: str):
    """Save cropped detections"""
    for idx, ((x1, y1, x2, y2), cls_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        crop_name = f"{image_stem}_cls{int(cls_id)}_{class_names[int(cls_id)]}_{idx}_{int(conf*100)}.jpg"
        cv2.imwrite(str(crops_dir / crop_name), crop)


def load_hand_coordinates(coords_file: Path) -> List[Dict]:
    """Load hand coordinates from the coordinates file"""
    hand_coords = []
    if coords_file.exists():
        with open(coords_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    # Handle both old format (12 columns) and new format (13 columns)
                    if len(parts) >= 12:
                        hand_coord = {
                            'image_name': parts[0],
                            'hand_id': int(parts[1]),
                            'bbox': (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])),
                            'center': (int(parts[6]), int(parts[7])),
                            'y_max': int(parts[8]),
                            'crossed_line_y': float(parts[9]),  # Changed to float for inclined lines
                            'frame_idx': int(parts[11] if len(parts) >= 13 else parts[10]),
                            'timestamp': parts[12] if len(parts) >= 13 else parts[11]
                        }
                        
                        # Add crossed_line_idx if available (new format)
                        if len(parts) >= 13:
                            hand_coord['crossed_line_idx'] = int(parts[10])
                        
                        hand_coords.append(hand_coord)
    return hand_coords


def apply_hand_bbox_padding(hand_bbox, padding, image_width, image_height):
    """Apply padding to hand bounding box while keeping within image bounds"""
    hand_x1, hand_y1, hand_x2, hand_y2 = hand_bbox
    
    # Apply padding
    padded_x1 = max(0, hand_x1 - padding)
    padded_y1 = max(0, hand_y1 - padding)
    padded_x2 = min(image_width, hand_x2 + padding)
    padded_y2 = min(image_height, hand_y2 + padding)
    
    return (padded_x1, padded_y1, padded_x2, padded_y2)


def check_intersection(obj_bbox, hand_bbox, threshold=0.1):
    """Check if object bounding box intersects with hand bounding box"""
    obj_x1, obj_y1, obj_x2, obj_y2 = obj_bbox
    hand_x1, hand_y1, hand_x2, hand_y2 = hand_bbox
    
    # Calculate intersection
    inter_x1 = max(obj_x1, hand_x1)
    inter_y1 = max(obj_y1, hand_y1)
    inter_x2 = min(obj_x2, hand_x2)
    inter_y2 = min(obj_y2, hand_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False, 0.0
    
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    obj_area = (obj_x2 - obj_x1) * (obj_y2 - obj_y1)
    hand_area = (hand_x2 - hand_x1) * (hand_y2 - hand_y1)
    
    # Calculate IoU
    union_area = obj_area + hand_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    # Check if intersection is significant
    return iou > threshold, iou


def process_hand_folder(hand_folder: Path, model, output_base_dir: Path, conf_threshold: float, save_crops: bool):
    """Process all images in a hand detection folder"""
    print(f"Processing hand folder: {hand_folder.name}")
    print(f"  Hand folder path: {hand_folder}")
    
    # Create corresponding output folder mirroring hand_detections structure:
    # objects_classification_output/<video_folder>/<hand_folder>
    # hand_folder is typically: .../<video_folder>/detected_hands/<hand_folder>
    try:
        video_folder_name = hand_folder.parents[1].name  # .../<video_folder>/detected_hands/<hand_folder>
    except Exception:
        video_folder_name = hand_folder.parent.name

    output_folder = output_base_dir / video_folder_name / hand_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load hand coordinates
    coords_file = hand_folder / "hand_coordinates.txt"
    print(f"  Video folder: {video_folder_name}")
    print(f"  Looking for coordinates file: {coords_file}")
    
    if not coords_file.exists():
        print(f"  Warning: hand_coordinates.txt not found in {hand_folder}")
        return
    
    hand_coords = load_hand_coordinates(coords_file)
    print(f"  Loaded {len(hand_coords)} hand coordinate entries")
    hand_coords_dict = {coord['image_name']: coord for coord in hand_coords}
    
    # Create CSV for object detections
    csv_path = output_folder / "object_detections.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
            "intersects_with_hand", "iou_with_hand", "hand_id", "hand_bbox", 
            "crossed_line_y", "crossed_line_idx"
        ])
        
        # Process each image in the folder
        image_files = [f for f in hand_folder.iterdir() if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        print(f"  Found {len(image_files)} image files to process")
        
        if not image_files:
            print(f"  Warning: No image files found in {hand_folder}")
            return
        
        for image_file in image_files:
            print(f"  Processing: {image_file.name}")
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"    Warning: Could not read image: {image_file}")
                continue
            
            # Run YOLO detection
            results = model(image, verbose=False, conf=conf_threshold)
            result = results[0]
            
            # Extract detections
            boxes_list = []
            class_ids_list = []
            confidences_list = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    boxes_list.append((x1, y1, x2, y2))
                    class_ids_list.append(cls_id)
                    confidences_list.append(conf)
            
            # Check intersections with hand (apply padding for better detection)
            hand_info = hand_coords_dict.get(image_file.name, {})
            original_hand_bbox = hand_info.get('bbox', (0, 0, 0, 0))
            hand_id = hand_info.get('hand_id', -1)
            crossed_line_y = hand_info.get('crossed_line_y', 0.0)
            crossed_line_idx = hand_info.get('crossed_line_idx', -1)
            
            # Apply padding to hand bbox for better intersection detection
            height, width = image.shape[:2]
            padded_hand_bbox = apply_hand_bbox_padding(original_hand_bbox, HAND_BBOX_PADDING, width, height)
            
            # Annotate image
            annotated = image.copy()
            class_names = result.names if hasattr(result, "names") else {}
            draw_detections(annotated, boxes_list, class_ids_list, confidences_list, class_names)
            
            # Draw original and padded hand bounding boxes for debugging
            if DRAW_PADDED_HAND_BBOX and original_hand_bbox != (0, 0, 0, 0):
                # Draw original hand bbox in blue
                cv2.rectangle(annotated, 
                            (original_hand_bbox[0], original_hand_bbox[1]), 
                            (original_hand_bbox[2], original_hand_bbox[3]), 
                            (255, 0, 0), 2)
                cv2.putText(annotated, f"Hand {hand_id}", 
                          (original_hand_bbox[0], original_hand_bbox[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw padded hand bbox in cyan
                cv2.rectangle(annotated, 
                            (padded_hand_bbox[0], padded_hand_bbox[1]), 
                            (padded_hand_bbox[2], padded_hand_bbox[3]), 
                            (255, 255, 0), 1)
                cv2.putText(annotated, f"Padded (+{HAND_BBOX_PADDING}px)", 
                          (padded_hand_bbox[0], padded_hand_bbox[3] + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Save annotated image
            annotated_name = output_folder / f"{image_file.stem}_annotated.jpg"
            cv2.imwrite(str(annotated_name), annotated)
            
            # Write detections to CSV
            for i, (bbox, cls_id, conf) in enumerate(zip(boxes_list, class_ids_list, confidences_list)):
                # Use padded bbox for intersection calculation with lower threshold
                intersects, iou = check_intersection(bbox, padded_hand_bbox, threshold=PADDED_IOU_THRESHOLD)
                
                # But store original hand bbox in CSV for reference
                writer.writerow([
                    image_file.name,
                    cls_id,
                    class_names.get(cls_id, str(cls_id)),
                    f"{conf:.4f}",
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    intersects,
                    f"{iou:.4f}",
                    hand_id,
                    f"{original_hand_bbox[0]},{original_hand_bbox[1]},{original_hand_bbox[2]},{original_hand_bbox[3]}",
                    f"{crossed_line_y:.4f}",
                    crossed_line_idx
                ])
            
            # Optionally save crops
            if save_crops and boxes_list:
                crops_dir = output_folder / "crops"
                crops_dir.mkdir(exist_ok=True)
                save_crops_for_image(image, boxes_list, class_ids_list, confidences_list, class_names, crops_dir, image_file.stem)
            
            print(f"    Found {len(boxes_list)} objects, saved to {annotated_name.name}")


def main():
    """Main function to process all hand detection folders"""
    print(f"Loading YOLO model: {MODEL}")
    model = YOLO(MODEL)
    
    # Setup input and output directories
    input_dir = Path(HAND_DETECTIONS_INPUT)
    output_dir = Path(OBJECTS_CLASSIFICATION_OUTPUT)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load analysis status to skip videos already classified
    analysis_status_path = Path(ANALYSIS_CSV)
    status = load_analysis_status(analysis_status_path)

    # Find all hand detection folders
    hand_folders = []
    for item in input_dir.iterdir():
        if item.is_dir():
            # Look for folders that contain detected_hands subfolder
            detected_hands_dir = item / "detected_hands"
            if detected_hands_dir.exists():
                # Process each subfolder in detected_hands
                for hand_subfolder in detected_hands_dir.iterdir():
                    if hand_subfolder.is_dir():
                        hand_folders.append(hand_subfolder)
    
    if not hand_folders:
        print(f"No hand detection folders found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(hand_folders)} hand detection folders to process")
    
    # Process each hand folder
    for hand_folder in hand_folders:
        try:
            # Determine video folder name for CSV key
            try:
                video_folder_name = hand_folder.parents[1].name
            except Exception:
                video_folder_name = hand_folder.parent.name

            # If object classification already done for this video, skip
            row = status.get(video_folder_name, {})
            if str(row.get("object_classification_done", "False")).lower() == "true":
                print(f"Skipping object classification for {video_folder_name} (already done)")
                continue

            process_hand_folder(hand_folder, model, output_dir, CONF_THRESHOLD, SAVE_CROPS)
        except Exception as e:
            print(f"Error processing {hand_folder}: {e}")
    
    # Mark video-level completion for all videos found
    for item in input_dir.iterdir():
        if item.is_dir():
            video_name = item.name
            row = status.get(video_name, {})
            row["video_name"] = video_name
            # Do not override hand step flag
            row.setdefault("hand_detection_done", True)  # likely true if folders exist
            row["object_classification_done"] = True
            row.setdefault("intersections_done", False)
            row["last_run"] = datetime.now().isoformat() if 'datetime' in globals() else ''
            status[video_name] = row

    try:
        write_analysis_status(analysis_status_path, status)
    except Exception as e:
        print(f"Warning: could not write analysis status CSV: {e}")

    print(f"\nObject classification complete!")
    print(f"Output directory: {output_dir}")
    print(f"Processed {len(hand_folders)} hand detection folders")


if __name__ == "__main__":
    main()
