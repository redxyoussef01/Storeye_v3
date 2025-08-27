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


def run_pipeline_once(python_executable: str, skip_crop: bool, skip_hands: bool, skip_objects: bool, skip_analysis: bool, exit_on_error: bool = True) -> None:
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
    else:
        print("Skipping intersection analysis step")

    print("\nPipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Storeyes end-to-end pipeline")
    parser.add_argument("--skip-crop", action="store_true", help="Skip video cropping step")
    parser.add_argument("--skip-hands", action="store_true", help="Skip hand detection step")
    parser.add_argument("--skip-objects", action="store_true", help="Skip object classification step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip intersection analysis step")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = parser.parse_args()
    
    print("Starting hourly loop. Press Ctrl+C to stop.")
    while True:
        start_time = time.time()
        try:
            run_pipeline_once(
                python_executable=args.python,
                skip_crop=args.skip_crop,
                skip_hands=args.skip_hands,
                skip_objects=args.skip_objects,
                skip_analysis=args.skip_analysis,
                exit_on_error=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Pipeline run encountered an error: {exc}")

        elapsed = time.time() - start_time
        sleep_seconds = max(0, 3600 - int(elapsed))
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
    """Delete all contents under data/cropped_videos after analysis finishes."""
    target_dir = os.path.join("data", "cropped_videos")
    if not os.path.isdir(target_dir):
        print("cropped_videos folder not found; nothing to clean")
        return
    removed_count = 0
    for name in os.listdir(target_dir):
        path = os.path.join(target_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to remove {path}: {exc}")
    print(f"cropped_videos cleaned: removed {removed_count} item(s)")


if __name__ == "__main__":
    main()


