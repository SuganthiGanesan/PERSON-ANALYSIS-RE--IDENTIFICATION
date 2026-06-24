"""
main.py
-------
Person Analysis & Re-Identification System — Main Entry Point

Usage:
    # Webcam (default)
    python main.py

    # Video file
    python main.py --source path/to/video.mp4

    # Change YOLO model size (n=fastest, s, m, l=best accuracy)
    python main.py --model s

    # Adjust re-ID sensitivity (0.0–1.0, lower = more aggressive matching)
    python main.py --reid-threshold 0.70

    # Run on image (single frame)
    python main.py --source path/to/image.jpg --image-mode
"""

import cv2
import time
import argparse
import os

from detector import PersonDetector
from colour_detector import DressColourDetector
from body_analyzer import BodyAnalyzer
from reid_tracker import PersonReID
from utils import draw_person, draw_dashboard, draw_no_detection
from typing import Optional


# ──────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Person Re-Identification System")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, or path to video/image file")
    parser.add_argument("--model", type=str, default="n",
                        choices=["n", "s", "m", "l"],
                        help="YOLOv8 model size (n=fast, l=accurate)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (0.0–1.0)")
    parser.add_argument("--reid-threshold", type=float, default=0.72,
                        help="Re-ID similarity threshold (0.0–1.0)")
    parser.add_argument("--image-mode", action="store_true",
                        help="Run on a single image instead of video")
    parser.add_argument("--no-body-analysis", action="store_true",
                        help="Disable MediaPipe body analysis (faster)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Core Processing
# ──────────────────────────────────────────────

def process_frame(
    frame,
    detector: PersonDetector,
    colour_det: DressColourDetector,
    body_analyzer: Optional[BodyAnalyzer],
    reid: PersonReID,
    stats: dict,
):
    """Process a single frame: detect → analyse → track → draw."""
    boxes = detector.detect(frame)

    if not boxes:
        draw_no_detection(frame)
        return frame

    for box in boxes:
        x1, y1, x2, y2, conf = box
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # --- Feature Extraction ---
        colour_name, _ = colour_det.get_dominant_colour(crop)

        if body_analyzer:
            body_info = body_analyzer.analyze(crop, (x1, y1, x2, y2), frame.shape[0])
            build = body_info["build"]
            height = body_info["height_estimate"]
        else:
            build = "N/A"
            height = "N/A"

        # --- Re-ID ---
        person_id, is_reid = reid.update(crop, colour_name, build, height)

        # --- Draw ---
        draw_person(
            frame=frame,
            box=(x1, y1, x2, y2),
            person_id=person_id,
            is_reid=is_reid,
            colour_name=colour_name,
            build=build,
            height=height,
            confidence=conf,
        )

    return frame


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────

def run_video(args):
    # Initialise components
    detector = PersonDetector(model_size=args.model, confidence=args.confidence)
    colour_det = DressColourDetector()
    body_analyzer = BodyAnalyzer() if not args.no_body_analysis else None
    reid = PersonReID(similarity_threshold=args.reid_threshold)

    # Open source
    source = int(args.source) if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        return

    print("\n[System] Running Person Re-ID System")
    print(f"  Source       : {args.source}")
    print(f"  YOLO model   : yolov8{args.model}")
    print(f"  ReID threshold: {args.reid_threshold}")
    print("  Press [Q] to quit, [S] to save screenshot\n")

    # Create resizable window at 1280x720
    WIN = "Person Re-Identification System"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    # FPS tracking
    fps = 0.0
    prev_time = time.time()
    frame_count = 0
    stats = {}
    screenshot_dir = "screenshots"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Info] Stream ended.")
            break

        frame_count += 1

        # Upscale small frames so labels have room to render
        fh, fw = frame.shape[:2]
        if fw < 960:
            scale = 960 / fw
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)))

        # --- Process ---
        frame = process_frame(frame, detector, colour_det, body_analyzer, reid, stats)

        # --- FPS ---
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = now

        # --- HUD ---
        draw_dashboard(frame, reid.get_stats(), fps)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("s"):
            os.makedirs(screenshot_dir, exist_ok=True)
            fname = os.path.join(screenshot_dir, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[Screenshot] Saved: {fname}")

    # --- Summary ---
    print("\n[Summary]")
    for pid, person in reid.get_all_persons().items():
        print(f"  Person {pid}: Dress={person.colour}, Build={person.build}, "
              f"Seen={person.seen_count}x, First={person.first_seen}, Last={person.last_seen}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Done]")


def run_image(args):
    """Process a single image."""
    img = cv2.imread(args.source)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.source}")
        return

    detector = PersonDetector(model_size=args.model, confidence=args.confidence)
    colour_det = DressColourDetector()
    body_analyzer = BodyAnalyzer() if not args.no_body_analysis else None
    reid = PersonReID(similarity_threshold=args.reid_threshold)

    result = process_frame(img, detector, colour_det, body_analyzer, reid, {})
    draw_dashboard(result, reid.get_stats(), fps=0.0)

    output_path = "output_image.jpg"
    cv2.imwrite(output_path, result)
    print(f"[Done] Result saved to {output_path}")

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    try:
        args = parse_args()
        if args.image_mode:
            run_image(args)
        else:
            run_video(args)
    except Exception:
        traceback.print_exc()
        input("\n[Crashed] Press Enter to exit...")