"""
Storeyes End-to-End Pipeline

This pipeline orchestrates the complete video analysis workflow:
1. Video cropping (motion detection)
2. Hand detection
3. Object classification
4. Intersection analysis
5. MongoDB summary upload (optional)

The pipeline can be run with various skip options to run only specific steps.
MongoDB integration automatically sends analysis results to MongoDB Atlas
after each successful analysis run.
"""

import argparse
import os
import sys
import subprocess
import time
import csv
from datetime import datetime, timezone
import shutil


def run_step(description: str, command: list[str], exit_on_error: bool = True) -> None:
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        if exit_on_error:
            sys.exit(result.returncode)
        raise RuntimeError(f"Step failed: {description} (exit {result.returncode})")


def ensure_directories() -> None:
    os.makedirs("data/cropped_videos", exist_ok=True)
    os.makedirs("detections_output", exist_ok=True)


def run_pipeline_once(python_executable: str, skip_crop: bool, skip_hands: bool, skip_objects: bool, skip_analysis: bool, skip_mongodb: bool, exit_on_error: bool = True) -> None:
    ensure_directories()

    if not skip_crop:
        run_step("Cropping videos", [python_executable, "video_crop.py"], exit_on_error=exit_on_error)
    else:
        print("Skipping cropping step")

    if not skip_hands:
        run_step("Hand detection", [python_executable, "hand_detection_fixed.py"], exit_on_error=exit_on_error)
    else:
        print("Skipping hand detection step")

    if not skip_objects:
        run_step("Object classification", [python_executable, "object_classification.py"], exit_on_error=exit_on_error)
    else:
        print("Skipping object classification step")

    if not skip_analysis:
        run_step("Intersection analysis", [python_executable, "analyze_intersections.py"], exit_on_error=exit_on_error)
        try:
            _mark_analysis_log_all_true()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to update analysis_log.csv after analysis: {exc}")
        try:
            _cleanup_cropped_videos()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to cleanup cropped videos: {exc}")
        
        # Send summary to MongoDB
        if not skip_mongodb:
            try:
                _send_summary_to_mongodb()
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to send summary to MongoDB: {exc}")
        else:
            print("Skipping MongoDB summary upload step")
    else:
        print("Skipping intersection analysis step")

    print("\nPipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Storeyes end-to-end pipeline")
    parser.add_argument("--skip-crop", action="store_true", help="Skip video cropping step")
    parser.add_argument("--skip-hands", action="store_true", help="Skip hand detection step")
    parser.add_argument("--skip-objects", action="store_true", help="Skip object classification step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip intersection analysis step")
    parser.add_argument("--skip-mongodb", action="store_true", help="Skip MongoDB summary upload step")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = parser.parse_args()
    
    print("Starting hourly loop. Press Ctrl+C to stop.")
    print("Available options:")
    print("  --skip-crop      : Skip video cropping step")
    print("  --skip-hands     : Skip hand detection step")
    print("  --skip-objects   : Skip object classification step")
    print("  --skip-analysis  : Skip intersection analysis step")
    print("  --skip-mongodb   : Skip MongoDB summary upload step")
    print()
    while True:
        start_time = time.time()
        try:
            run_pipeline_once(
                python_executable=args.python,
                skip_crop=args.skip_crop,
                skip_hands=args.skip_hands,
                skip_objects=args.skip_objects,
                skip_analysis=args.skip_analysis,
                skip_mongodb=args.skip_mongodb,
                exit_on_error=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Pipeline run encountered an error: {exc}")

        elapsed = time.time() - start_time
        sleep_seconds = max(0, 20 - int(elapsed))
        print(f"Waiting {sleep_seconds} seconds before next run...")
        time.sleep(sleep_seconds)


def _mark_analysis_log_all_true() -> None:
    """Set all boolean status columns to True in detections_output/analysis_log.csv.

    Columns expected: video_name, hand_detection_done, object_classification_done, intersections_done, last_run
    If the file does not exist, do nothing.
    """
    csv_path = os.path.join("detections_output", "analysis_log.csv")
    if not os.path.exists(csv_path):
        print("analysis_log.csv not found; skipping status update")
        return

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        expected = {"video_name", "hand_detection_done", "object_classification_done", "intersections_done", "last_run"}
        missing = expected.difference(fieldnames)
        if missing:
            print(f"analysis_log.csv missing columns {missing}; skipping status update")
            return
        for row in reader:
            row["hand_detection_done"] = "True"
            row["object_classification_done"] = "True"
            row["intersections_done"] = "True"
            row["last_run"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["video_name", "hand_detection_done", "object_classification_done", "intersections_done", "last_run"])
        writer.writeheader()
        writer.writerows(rows)
    print("analysis_log.csv updated: all status columns set to True")


def _cleanup_cropped_videos() -> None:
    """Delete all contents under data/cropped_videos after analysis finishes, but preserve CSV files."""
    target_dir = os.path.join("data", "cropped_videos")
    if not os.path.isdir(target_dir):
        print("cropped_videos folder not found; nothing to clean")
        return
    removed_count = 0
    for name in os.listdir(target_dir):
        path = os.path.join(target_dir, name)
        try:
            # Skip CSV files to preserve them
            if name.lower().endswith('.csv'):
                print(f"Preserving CSV file: {name}")
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to remove {path}: {exc}")
    print(f"cropped_videos cleaned: removed {removed_count} item(s) (CSV files preserved)")


def _send_summary_to_mongodb() -> None:
    """Send analysis summary to MongoDB after analysis completes."""
    try:
        print("\n📤 Sending analysis summary to MongoDB...")
        
        # Import the MongoDB sending function
        from send_summary_to_mongodb import load_summary_data, prepare_mongodb_document, connect_to_mongodb, insert_to_mongodb, create_indexes
        
        # Load summary data
        summary_json_path = "./detections_output/objects_classification_output/global_summary.json"
        summary_data = load_summary_data(summary_json_path)
        if not summary_data:
            print("⚠️  Warning: No summary data found to send to MongoDB")
            return
        
        # Connect to MongoDB
        client, db, collection = connect_to_mongodb()
        if not client:
            print("⚠️  Warning: Could not connect to MongoDB")
            return
        
        try:
            # Prepare document for MongoDB
            document = prepare_mongodb_document(summary_data)
            
            # Insert into MongoDB
            success, document_id = insert_to_mongodb(collection, document)
            
            if success:
                print(f"✅ Successfully sent analysis summary to MongoDB!")
                print(f"   Document ID: {document_id}")
                
                # Create indexes for better performance
                create_indexes(collection)
                
            else:
                print("⚠️  Warning: Failed to send analysis summary to MongoDB")
                
        finally:
            # Close MongoDB connection
            client.close()
            
    except ImportError as e:
        print(f"⚠️  Warning: Could not import MongoDB functions: {e}")
        print("   Make sure send_summary_to_mongodb.py is available")
    except Exception as e:
        print(f"⚠️  Warning: Error sending summary to MongoDB: {e}")


if __name__ == "__main__":
    main()


