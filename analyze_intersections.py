import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# ---- Configuration (edit these variables) ----
# Input directory (object classification outputs)
OBJECTS_CLASSIFICATION_INPUT: str = "detections_output/objects_classification_output"

# Shared analysis CSV path
ANALYSIS_CSV: str = "detections_output/analysis_log.csv"

# IoU threshold for considering intersection significant (matching object_classification.py)
IOU_THRESHOLD: float = 0.05


def parse_video_metadata(video_folder_name: str) -> Dict:
    """Parse video folder name to extract metadata"""
    try:
        # Format: chunk_XX_YYYYMMDD_HHMMSS
        parts = video_folder_name.split('_')
        if len(parts) >= 4:
            chunk_number = parts[1]
            date_str = parts[2]  # YYYYMMDD
            time_str = parts[3]  # HHMMSS
            
            # Convert to datetime
            datetime_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            
            return {
                "video_name": video_folder_name,
                "chunk_number": chunk_number,
                "start_date": datetime_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "start_date_iso": datetime_obj.isoformat(),
                "date": date_str,
                "time": time_str
            }
    except Exception as e:
        print(f"Warning: Could not parse video metadata from '{video_folder_name}': {e}")
    
    # Fallback
    return {
        "video_name": video_folder_name,
        "chunk_number": "unknown",
        "start_date": "unknown",
        "start_date_iso": "unknown",
        "date": "unknown",
        "time": "unknown"
    }


def create_hand_json_summary(results: Dict, hand_folder_path: Path, video_metadata: Dict):
    """Create JSON summary for a hand folder"""
    # Extract hand-specific data
    hand_folder_name = hand_folder_path.name
    
    print(f"    Creating JSON summary for hand folder: {hand_folder_name}")
    print(f"    Hand folder path: {hand_folder_path}")
    print(f"    Hand folder exists: {hand_folder_path.exists()}")
    
    # Create JSON structure
    json_data = {
        "video_name": video_metadata["video_name"],
        "start_date": video_metadata["start_date"],
        "start_date_iso": video_metadata["start_date_iso"],
        "chunk_number": video_metadata["chunk_number"],
        "hand_folder": hand_folder_name,
        "output_objects": dict(results.get('folder_class_distribution', {})),
        "details": {
            "total_objects_frequent_count": results.get('folder_object_count', 0),
            "total_raw_detections": results.get('total_objects', 0),
            "images_processed": results.get('images_total', 0),
            "images_with_objects": results.get('images_with_objects', 0),
            "average_objects_per_frame": results.get('avg_objects_per_frame', 0.0),
            "intersection_rate_percent": results.get('intersection_rate', 0.0),
            "average_confidence": results.get('folder_avg_confidence', 0.0),
            "hand_counts": dict(results.get('folder_hand_counts', {})),
            "hand_frequent_objects": dict(results.get('folder_hand_objects', {})),
            "confidence_distribution": dict(results.get('confidence_ranges', {})),
            "iou_distribution": dict(results.get('iou_ranges', {})),
            "frame_object_counts": results.get('frame_object_counts', [])
        },
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "iou_threshold": IOU_THRESHOLD,
            "analysis_method": "frequency_based_counting"
        }
    }
    
    # Save JSON file
    json_file = hand_folder_path / f"{hand_folder_name}_summary.json"
    print(f"    JSON file path: {json_file}")
    print(f"    JSON file absolute path: {json_file.absolute()}")
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"    Successfully created JSON summary: {json_file}")
        print(f"    File size: {json_file.stat().st_size} bytes")
        return json_data
    except Exception as e:
        print(f"    Error writing JSON file: {e}")
        raise


def load_csv_data(csv_file: Path) -> List[Dict]:
    """Load data from CSV file"""
    data = []
    if csv_file.exists():
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                row['class_id'] = int(row['class_id'])
                row['confidence'] = float(row['confidence'])
                row['x1'] = int(row['x1'])
                row['y1'] = int(row['y1'])
                row['x2'] = int(row['x2'])
                row['y2'] = int(row['y2'])
                row['intersects_with_hand'] = row['intersects_with_hand'].lower() == 'true'
                row['iou_with_hand'] = float(row['iou_with_hand'])
                row['hand_id'] = int(row['hand_id'])
                
                # Handle new columns if they exist
                if 'crossed_line_y' in row:
                    row['crossed_line_y'] = float(row['crossed_line_y'])
                if 'crossed_line_idx' in row:
                    row['crossed_line_idx'] = int(row['crossed_line_idx'])
                    
                data.append(row)
    return data


def analyze_folder(folder_path: Path) -> Dict:
    """Analyze all CSV files in a folder"""
    print(f"Analyzing folder: {folder_path.name}")
    
    # Find only the primary classification CSV to avoid re-processing generated CSVs
    primary_csv = folder_path / "object_detections.csv"
    csv_files = [primary_csv] if primary_csv.exists() else []
    
    if not csv_files:
        print(f"  No object_detections.csv found in {folder_path}")
        return {}
    
    # Statistics containers (raw)
    total_objects = 0
    intersecting_objects = 0
    class_statistics = Counter()
    hand_statistics = Counter()
    confidence_ranges = defaultdict(int)
    iou_ranges = defaultdict(int)
    
    # Detailed data for output (raw)
    all_detections = []
    intersecting_detections = []

    # Group detections by image for per-image majority selection
    detections_by_image: Dict[str, List[Dict]] = defaultdict(list)
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"  Processing: {csv_file.name}")
        data = load_csv_data(csv_file)
        
        for row in data:
            total_objects += 1
            all_detections.append(row)
            detections_by_image[row['image_name']].append(row)
            
            # Class statistics
            class_statistics[row['class_name']] += 1
            
            # Hand statistics
            hand_statistics[row['hand_id']] += 1
            
            # Confidence range
            conf = row['confidence']
            if conf >= 0.9:
                confidence_ranges['0.9-1.0'] += 1
            elif conf >= 0.8:
                confidence_ranges['0.8-0.9'] += 1
            elif conf >= 0.7:
                confidence_ranges['0.7-0.8'] += 1
            elif conf >= 0.6:
                confidence_ranges['0.6-0.7'] += 1
            else:
                confidence_ranges['0.0-0.6'] += 1
            
            # Check for intersection
            if row['intersects_with_hand'] and row['iou_with_hand'] >= IOU_THRESHOLD:
                intersecting_objects += 1
                intersecting_detections.append(row)
                
                # IoU range for intersecting objects
                iou = row['iou_with_hand']
                if iou >= 0.5:
                    iou_ranges['0.5-1.0'] += 1
                elif iou >= 0.3:
                    iou_ranges['0.3-0.5'] += 1
                elif iou >= 0.1:
                    iou_ranges['0.1-0.3'] += 1
                else:
                    iou_ranges['0.0-0.1'] += 1
    
    # IMPROVED COUNTING LOGIC: Find most frequent objects per hand across all images
    # Step 1: Group intersecting objects by hand_id and image for frequency analysis
    hand_image_objects: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    
    for image_name, dets in detections_by_image.items():
        # Get only intersecting detections above IoU threshold
        intersecting_in_frame = [d for d in dets if d.get('intersects_with_hand') and d.get('iou_with_hand', 0.0) >= IOU_THRESHOLD]
        
        # Group by hand_id for this image
        for obj in intersecting_in_frame:
            hand_id = obj['hand_id']
            hand_image_objects[hand_id][image_name].append(obj)
    
    # Step 2: For each hand, count objects by class in each image, then find most frequent count
    folder_hand_counts = {}  # hand_id -> most frequent object count
    folder_hand_class_counts = {}  # hand_id -> {class_name: most_frequent_count}
    folder_total_objects = 0
    best_frame_objects = []  # Representative objects for this folder
    
    print(f"  Found hands: {list(hand_image_objects.keys())}")
    
    for hand_id, images_data in hand_image_objects.items():
        print(f"    Analyzing hand {hand_id} across {len(images_data)} images")
        
        # Count objects by class in each image for this hand
        hand_class_frame_counts = defaultdict(list)  # class_name -> [count_frame1, count_frame2, ...]
        total_objects_per_frame = []
        
        for image_name, objects in images_data.items():
            # Count by class in this specific image for this hand
            image_class_counts = Counter()
            for obj in objects:
                image_class_counts[obj['class_name']] += 1
            
            # Record counts for each class
            for class_name, count in image_class_counts.items():
                hand_class_frame_counts[class_name].append(count)
            
            # For classes not present in this frame, add 0
            all_classes_for_hand = set()
            for img_objs in images_data.values():
                for obj in img_objs:
                    all_classes_for_hand.add(obj['class_name'])
            
            for class_name in all_classes_for_hand:
                if class_name not in image_class_counts:
                    hand_class_frame_counts[class_name].append(0)
            
            total_objects_per_frame.append(len(objects))
        
        # Find most frequent count for each class for this hand
        hand_class_frequent_counts = {}
        hand_total_frequent_count = 0
        
        for class_name, frame_counts in hand_class_frame_counts.items():
            if frame_counts:
                # Find the most frequent count across all frames for this class
                count_frequency = Counter(frame_counts)
                most_frequent_count = count_frequency.most_common(1)[0][0]
                hand_class_frequent_counts[class_name] = most_frequent_count
                hand_total_frequent_count += most_frequent_count
                
                print(f"      Class {class_name}: frame counts {frame_counts} → frequent count: {most_frequent_count}")
        
        # Store results for this hand
        folder_hand_counts[hand_id] = hand_total_frequent_count
        folder_hand_class_counts[hand_id] = hand_class_frequent_counts
        folder_total_objects += hand_total_frequent_count
        
        # Add representative objects for this hand (highest confidence objects for each frequent class)
        for class_name, frequent_count in hand_class_frequent_counts.items():
            if frequent_count > 0:
                # Find highest confidence objects of this class from this hand
                all_hand_objects = []
                for objects in images_data.values():
                    all_hand_objects.extend([obj for obj in objects if obj['class_name'] == class_name])
                
                if all_hand_objects:
                    # Sort by confidence and take top objects
                    sorted_objects = sorted(all_hand_objects, key=lambda x: x['confidence'], reverse=True)
                    for i in range(min(frequent_count, len(sorted_objects))):
                        best_frame_objects.append(sorted_objects[i])
    
    # Per-image analysis for backward compatibility
    per_image_analysis: List[Dict] = []
    max_objects_in_frame = 0
    best_frame_name = f"Frequent_objects_from_{len(folder_hand_counts)}_hands"
    frame_object_counts = []

    for image_name, dets in detections_by_image.items():
        # Get ALL intersecting detections above IoU threshold for this frame
        intersecting_in_frame = [d for d in dets if d.get('intersects_with_hand') and d.get('iou_with_hand', 0.0) >= IOU_THRESHOLD]
        
        # Group objects by hand_id to count objects per hand
        objects_per_hand = defaultdict(list)
        for obj in intersecting_in_frame:
            objects_per_hand[obj['hand_id']].append(obj)
        
        # Count total objects across all hands in this frame
        total_objects_in_frame = sum(len(hand_objects) for hand_objects in objects_per_hand.values())
        frame_object_counts.append(total_objects_in_frame)
        
        # Track frame with maximum objects
        if total_objects_in_frame > max_objects_in_frame:
            max_objects_in_frame = total_objects_in_frame
        
        # Count by class for this frame
        frame_class_counter = Counter()
        for obj in intersecting_in_frame:
            frame_class_counter[obj['class_name']] += 1
        
        # Count by hand for this frame
        frame_hand_counter = {}
        for hand_id, hand_objects in objects_per_hand.items():
            frame_hand_counter[hand_id] = len(hand_objects)
        
        # Build per-image record
        rec = {
            'image_name': image_name,
            'total_intersecting_objects': total_objects_in_frame,
            'objects_per_hand': dict(frame_hand_counter),
            'class_counts': dict(frame_class_counter),
            'objects_list': intersecting_in_frame
        }
        per_image_analysis.append(rec)
    
    # Calculate class distribution and average confidence from the new counting logic
    folder_class_distribution = Counter()
    folder_avg_confidence = 0.0
    total_conf = 0.0
    total_objects_for_avg = 0
        
    # Aggregate class counts from all hands
    for hand_id, hand_class_counts in folder_hand_class_counts.items():
        for class_name, count in hand_class_counts.items():
            folder_class_distribution[class_name] += count
    
    # Calculate average confidence from best frame objects
    if best_frame_objects:
        total_conf = sum(obj['confidence'] for obj in best_frame_objects)
        total_objects_for_avg = len(best_frame_objects)
        folder_avg_confidence = total_conf / total_objects_for_avg
    
    # Create folder_hand_objects for backward compatibility (most frequent object type per hand)
    folder_hand_objects = {}
    for hand_id, hand_class_counts in folder_hand_class_counts.items():
        if hand_class_counts:
            # Find the class with highest frequent count for this hand
            most_frequent_class = max(hand_class_counts.items(), key=lambda x: x[1])
            folder_hand_objects[hand_id] = most_frequent_class[0]
    
    # Calculate statistics for all frames
    images_with_objects = len([count for count in frame_object_counts if count > 0])
    avg_objects_per_frame = sum(frame_object_counts) / len(frame_object_counts) if frame_object_counts else 0

    # Calculate percentages (raw)
    intersection_rate = (intersecting_objects / total_objects * 100) if total_objects > 0 else 0
    
    # Compile results
    results = {
        'folder_name': folder_path.name,
        'total_objects': total_objects,
        'intersecting_objects': intersecting_objects,
        'intersection_rate': intersection_rate,
        'class_statistics': dict(class_statistics),
        'hand_statistics': dict(hand_statistics),
        'confidence_ranges': dict(confidence_ranges),
        'iou_ranges': dict(iou_ranges),
        'all_detections': all_detections,
        'intersecting_detections': intersecting_detections,
        # NEW: Improved frequency-based counting per hand
        'folder_object_count': folder_total_objects,
        'best_frame_name': best_frame_name,
        'best_frame_objects': best_frame_objects,
        'folder_hand_counts': dict(folder_hand_counts),  # Save frequent counts per hand
        'folder_hand_objects': dict(folder_hand_objects),  # Save most frequent object per hand
        'folder_hand_class_counts': dict(folder_hand_class_counts),  # Save detailed class counts per hand
        'folder_class_distribution': dict(folder_class_distribution),
        'folder_avg_confidence': folder_avg_confidence,
        'images_total': len(detections_by_image),
        'images_with_objects': images_with_objects,
        'avg_objects_per_frame': avg_objects_per_frame,
        'frame_object_counts': frame_object_counts,
        'per_image_analysis': per_image_analysis,
    }
    
    print(f"  Total objects (raw): {total_objects}")
    print(f"  Images with objects: {images_with_objects}/{len(detections_by_image)}")
    print(f"  Frequent objects count (folder): {folder_total_objects}")
    if folder_hand_counts:
        print(f"  Hand frequent counts: {dict(folder_hand_counts)}")
    if folder_hand_objects:
        print(f"  Hand frequent objects: {dict(folder_hand_objects)}")
    if folder_class_distribution:
        print(f"  Class distribution (frequent): {dict(folder_class_distribution)}")
        print(f"  Average confidence: {folder_avg_confidence:.3f}")
    print(f"  Average objects per frame: {avg_objects_per_frame:.2f}")
    print(f"  Intersection rate (raw detections): {intersection_rate:.2f}%")
    
    return results


def save_detailed_results(results: Dict, output_folder: Path):
    """Save detailed analysis results"""
    folder_name = results['folder_name']
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Save summary statistics (includes max-objects-per-frame count)
    summary_file = output_folder / f"{folder_name}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"INTERSECTION ANALYSIS SUMMARY\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Folder: {folder_name}\n")
        f.write(f"Total detections (raw rows): {results['total_objects']}\n")
        f.write(f"Images total: {results['images_total']}\n")
        f.write(f"Images with objects: {results['images_with_objects']}\n")
        f.write(f"\nFREQUENCY-BASED COUNTING (DUPLICATE FRAME HANDLING):\n")
        f.write(f"-" * 50 + "\n")
        f.write(f"Frequent objects count (folder): {results['folder_object_count']}\n")
        f.write(f"Representative frame: {results['best_frame_name']}\n")
        if 'folder_hand_counts' in results and results['folder_hand_counts']:
            f.write(f"Hand frequent counts (most common per hand):\n")
            for hand_id, frequent_count in results.get('folder_hand_counts', {}).items():
                # Get the most frequent object for this hand
                frequent_object = results.get('folder_hand_objects', {}).get(hand_id, 'unknown')
                f.write(f"  - Hand {hand_id}: {frequent_count} objects (most frequent across frames) - frequent object: {frequent_object}\n")
        if results['folder_class_distribution']:
            f.write(f"Class distribution (frequent counts):\n")
            for class_name, count in results['folder_class_distribution'].items():
                f.write(f"  - {class_name}: {count}\n")
            f.write(f"Average confidence: {results['folder_avg_confidence']:.3f}\n")
        f.write(f"Average objects per frame: {results['avg_objects_per_frame']:.2f}\n")
        f.write(f"Frame object counts: {results['frame_object_counts']}\n")
        f.write(f"Intersection rate (raw): {results['intersection_rate']:.2f}%\n\n")
        
        f.write(f"CLASS STATISTICS:\n")
        f.write(f"-" * 20 + "\n")
        for class_name, count in results['class_statistics'].items():
            f.write(f"{class_name}: {count} objects\n")
        
        f.write(f"\nHAND STATISTICS:\n")
        f.write(f"-" * 20 + "\n")
        for hand_id, count in results['hand_statistics'].items():
            f.write(f"Hand {hand_id}: {count} objects\n")
        
        f.write(f"\nCONFIDENCE RANGES:\n")
        f.write(f"-" * 20 + "\n")
        for range_name, count in results['confidence_ranges'].items():
            f.write(f"{range_name}: {count} objects\n")
        
        f.write(f"\nIoU RANGES (Intersecting Objects):\n")
        f.write(f"-" * 30 + "\n")
        for range_name, count in results['iou_ranges'].items():
            f.write(f"{range_name}: {count} objects\n")
    
    # 2. Save all detections CSV
    all_detections_file = output_folder / f"{folder_name}_all_detections.csv"
    if results['all_detections']:
        with open(all_detections_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = results['all_detections'][0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['all_detections'])
    
    # 3. Save intersecting detections CSV
    intersecting_file = output_folder / f"{folder_name}_intersecting_detections.csv"
    if results['intersecting_detections']:
        with open(intersecting_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = results['intersecting_detections'][0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['intersecting_detections'])

    # 4. Save per-frame analysis CSV (NEW: shows object counts per frame)
    per_frame_file = output_folder / f"{folder_name}_per_frame_analysis.csv"
    if results['per_image_analysis']:
        with open(per_frame_file, 'w', newline='', encoding='utf-8') as f:
            # Create simplified records for CSV (without objects_list)
            csv_records = []
            for frame in results['per_image_analysis']:
                record = {
                    'image_name': frame['image_name'],
                    'total_intersecting_objects': frame['total_intersecting_objects'],
                    'objects_per_hand': str(frame['objects_per_hand']),  # Convert dict to string for CSV
                    'class_counts': str(frame['class_counts'])  # Convert dict to string for CSV
                }
                csv_records.append(record)
            
            if csv_records:
                fieldnames = csv_records[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_records)
    
    # 5. Save best frame detections CSV (objects from the frame with max count)
    best_frame_file = output_folder / f"{folder_name}_best_frame_objects.csv"
    if results['best_frame_objects']:
        with open(best_frame_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = results['best_frame_objects'][0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['best_frame_objects'])
    
    print(f"  Saved detailed results to: {output_folder}")


def main():
    """Main function to analyze all object classification folders"""
    print(f"Starting intersection analysis...")
    
    # Setup input and output directories
    input_dir = Path(OBJECTS_CLASSIFICATION_INPUT)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Load and update analysis CSV to skip already analyzed videos
    status = {}
    csv_path = Path(ANALYSIS_CSV)
    if csv_path.exists():
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status[row.get("video_name", "")] = row
        except Exception as e:
            print(f"Warning: could not read analysis CSV: {e}")
    
    # Find all folders to analyze (new nested structure):
    # objects_classification_output/<video_folder>/<hand_folder>/object_detections.csv
    folders_to_analyze_set = set()
    for csv_path in input_dir.rglob("object_detections.csv"):
        folders_to_analyze_set.add(csv_path.parent)
    # Backward compatibility: also include any immediate subfolders that contain CSVs
    for item in input_dir.iterdir():
        if item.is_dir() and list(item.glob("object_detections.csv")):
            folders_to_analyze_set.add(item)
    folders_to_analyze = sorted(folders_to_analyze_set)
    
    if not folders_to_analyze:
        print(f"No folders with object_detections.csv found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(folders_to_analyze)} folders to analyze")
    
    # Organize folders by video
    video_folders = defaultdict(list)
    for folder in folders_to_analyze:
        try:
            # Determine video folder name
            # folder structure: objects_classification_output/<video_folder>/<hand_folder>/
            if folder.parent.name != input_dir.name:  # It's nested
                video_folder_name = folder.parent.name
            else:  # It's flat (backward compatibility)
                video_folder_name = "unknown_video"
            
            # Skip if analysis already marked done
            if str(status.get(video_folder_name, {}).get("intersections_done", "False")).lower() == "true":
                print(f"Skipping analysis for {video_folder_name} (already done)")
                continue

            video_folders[video_folder_name].append(folder)
        except Exception as e:
            print(f"Error organizing folder {folder}: {e}")
    
    print(f"Found {len(video_folders)} video folders: {list(video_folders.keys())}")
    
    # Global statistics (frame-based counting)
    global_folders = 0
    global_total_objects = 0
    global_class_stats = Counter()
    folders_with_objects = 0
    
    # Process each video folder
    all_results = []
    video_summaries = {}
    
    for video_name, hand_folders in video_folders.items():
        print(f"\nProcessing video: {video_name}")
        
        # Parse video metadata
        video_metadata = parse_video_metadata(video_name)
        
        # Video-level statistics
        video_folders_count = 0
        video_total_objects = 0
        video_class_stats = Counter()
        video_folders_with_objects = 0
        video_hand_results = []
        
        # Process each hand folder in this video
        for hand_folder in hand_folders:
            try:
                results = analyze_folder(hand_folder)
                if results:
                    all_results.append(results)
                    video_hand_results.append(results)
                    
                    # Accumulate video-level statistics
                    video_folders_count += 1
                    folder_count = results['folder_object_count']
                    video_total_objects += folder_count
                    
                    if folder_count > 0:
                        video_folders_with_objects += 1
                        # Add class distribution from best frame
                        for class_name, count in results['folder_class_distribution'].items():
                            video_class_stats[class_name] += count
                    
                    # Accumulate global statistics
                    global_folders += 1
                    global_total_objects += folder_count
                    
                    if folder_count > 0:
                        folders_with_objects += 1
                        for class_name, count in results['folder_class_distribution'].items():
                            global_class_stats[class_name] += count

                    # Save detailed results INSIDE the same hand folder
                    save_detailed_results(results, hand_folder)
                    
                    # Create JSON summary for this hand folder
                    try:
                        json_summary = create_hand_json_summary(results, hand_folder, video_metadata)
                        print(f"  Successfully created JSON summary for {hand_folder.name}")
                    except Exception as e:
                        print(f"  Error creating JSON summary for {hand_folder.name}: {e}")
                    
            except Exception as e:
                print(f"Error processing hand folder {hand_folder}: {e}")
        
        # Save video-level summary
        if video_hand_results:
            video_summary_data = {
                "video_metadata": video_metadata,
                "total_hand_folders": video_folders_count,
                "hand_folders_with_objects": video_folders_with_objects,
                "total_objects_count": video_total_objects,
                "average_objects_per_hand_folder": video_total_objects / video_folders_count if video_folders_count > 0 else 0,
                "class_distribution": dict(video_class_stats),
                "hand_folder_results": [
                    {
                        "folder_name": r['folder_name'],
                        "object_count": r['folder_object_count'],
                        "class_distribution": dict(r['folder_class_distribution']),
                        "hand_counts": dict(r.get('folder_hand_counts', {})),
                        "hand_objects": dict(r.get('folder_hand_objects', {}))
                    }
                    for r in video_hand_results
                ]
            }
            
            video_summaries[video_name] = video_summary_data
            
            # Save video-level global summary
            video_folder_path = input_dir / video_name
            if video_folder_path.exists():
                video_summary_file = video_folder_path / "global_summary.txt"
                with open(video_summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"VIDEO SUMMARY: {video_name}\n")
                    f.write(f"=" * 50 + "\n")
                    f.write(f"Video start date: {video_metadata['start_date']}\n")
                    f.write(f"Chunk number: {video_metadata['chunk_number']}\n")
                    f.write(f"Total hand folders processed: {video_folders_count}\n")
                    f.write(f"Hand folders with objects: {video_folders_with_objects}\n")
                    f.write(f"Total objects counted (frequent): {video_total_objects}\n")
                    f.write(f"Average objects per hand folder: {video_total_objects / video_folders_count if video_folders_count > 0 else 0:.2f}\n")
                    f.write(f"\nClass distribution:\n")
                    for class_name, count in video_class_stats.most_common():
                        f.write(f"- {class_name}: {count}\n")
                    f.write(f"\nPer-hand-folder results:\n")
                    f.write(f"-" * 30 + "\n")
                    for result in video_hand_results:
                        f.write(f"{result['folder_name']}: {result['folder_object_count']} objects")
                        if result['folder_class_distribution']:
                            f.write(f" ({dict(result['folder_class_distribution'])})")
                        if result.get('folder_hand_counts'):
                            f.write(f" [hands: {dict(result['folder_hand_counts'])}]")
                        if result.get('folder_hand_objects'):
                            f.write(f" [hand objects: {dict(result['folder_hand_objects'])}]")
                        f.write(f"\n")
                
                print(f"  Saved video summary: {video_summary_file}")
        
        print(f"Video {video_name}: {video_total_objects} total objects, {video_folders_with_objects}/{video_folders_count} folders with objects")

        # Mark analysis completion in CSV for this processed video
        row = status.get(video_name, {})
        row["video_name"] = video_name
        row.setdefault("hand_detection_done", True)
        row.setdefault("object_classification_done", True)
        row["intersections_done"] = True
        row["last_run"] = datetime.now().isoformat()
        status[video_name] = row

    # Also ensure any video folders present in objects classification output are marked as completed
    try:
        for item in input_dir.iterdir():
            if item.is_dir():
                vname = item.name
                row = status.get(vname, {})
                row["video_name"] = vname
                row.setdefault("hand_detection_done", True)
                row.setdefault("object_classification_done", True)
                # Force intersections_done to True if analysis reached this point
                row["intersections_done"] = True
                row["last_run"] = datetime.now().isoformat()
                status[vname] = row
    except Exception as e:
        print(f"Warning: could not ensure completion flags for all videos: {e}")

    # Save updated CSV
    try:
        fieldnames = [
            "video_name",
            "hand_detection_done",
            "object_classification_done",
            "intersections_done",
            "last_run",
        ]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for vname, row in sorted(status.items()):
                writer.writerow({
                    "video_name": vname,
                    "hand_detection_done": str(row.get("hand_detection_done", False)),
                    "object_classification_done": str(row.get("object_classification_done", False)),
                    "intersections_done": str(row.get("intersections_done", False)),
                    "last_run": row.get("last_run", "")
                })
    except Exception as e:
        print(f"Warning: could not write analysis CSV: {e}")
    
    # Save global summary (frame-based)
    if all_results:
        global_summary_file = input_dir / "global_summary.txt"
        with open(global_summary_file, 'w', encoding='utf-8') as f:
            f.write(f"GLOBAL INTERSECTION ANALYSIS SUMMARY (Frequency-based counting)\n")
            f.write(f"=" * 65 + "\n")
            f.write(f"Total hand folders processed: {global_folders}\n")
            f.write(f"Folders with objects: {folders_with_objects}\n")
            f.write(f"Total objects counted (frequent per folder): {global_total_objects}\n")
            f.write(f"Average objects per folder: {global_total_objects / global_folders:.2f}\n")
            f.write(f"\nNOTE: Uses frequency-based counting to handle duplicate frames.\n")
            f.write(f"For each hand, counts the most frequent number of objects across all frames.\n")
            f.write(f"\nClass distribution (frequent counts):\n")
            for class_name, count in global_class_stats.most_common():
                f.write(f"- {class_name}: {count}\n")
            f.write(f"\nPer-folder results:\n")
            f.write(f"-" * 30 + "\n")
            for result in all_results:
                f.write(f"{result['folder_name']}: {result['folder_object_count']} objects")
                if result['folder_class_distribution']:
                    f.write(f" ({dict(result['folder_class_distribution'])})")
                if result.get('folder_hand_counts'):
                    f.write(f" [hands: {dict(result['folder_hand_counts'])}]")
                if result.get('folder_hand_objects'):
                    f.write(f" [hand objects: {dict(result['folder_hand_objects'])}]")
                f.write(f"\n")
        print(f"\nGlobal summary saved to: {global_summary_file}")
        
        # Create comprehensive JSON summary
        global_json_summary = {
            "analysis_summary": {
                "total_hand_folders_processed": global_folders,
                "folders_with_objects": folders_with_objects,
                "total_objects_counted": global_total_objects,
                "average_objects_per_folder": round(global_total_objects / global_folders, 2) if global_folders > 0 else 0,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_method": "frequency_based_counting",
                "iou_threshold": IOU_THRESHOLD
            },
            "class_distribution": dict(global_class_stats),
            "video_summaries": video_summaries,
            "detailed_results": []
        }
        
        # Add detailed results for each folder
        for result in all_results:
            folder_detail = {
                "folder_name": result['folder_name'],
                "object_count": result['folder_object_count'],
                "class_distribution": dict(result['folder_class_distribution']),
                "hand_counts": dict(result.get('folder_hand_counts', {})),
                "hand_objects": dict(result.get('folder_hand_objects', {})),
                "images_total": result['images_total'],
                "images_with_objects": result['images_with_objects'],
                "average_objects_per_frame": result['avg_objects_per_frame'],
                "intersection_rate": result['intersection_rate'],
                "average_confidence": result['folder_avg_confidence']
            }
            global_json_summary["detailed_results"].append(folder_detail)
        
        # Save comprehensive JSON summary
        global_json_file = input_dir / "global_summary.json"
        with open(global_json_file, 'w', encoding='utf-8') as f:
            json.dump(global_json_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive JSON summary saved to: {global_json_file}")
        print(f"JSON file size: {global_json_file.stat().st_size} bytes")
    
    print(f"\nIntersection analysis complete!")
    print(f"Processed {len(all_results)} folders")
    print(f"Total objects counted (frequency-based): {global_total_objects}")
    print(f"Folders with objects: {folders_with_objects}/{global_folders}")


if __name__ == "__main__":
    main()
