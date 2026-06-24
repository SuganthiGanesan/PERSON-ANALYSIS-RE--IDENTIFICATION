"""
detector.py
-----------
Person detection using YOLOv8.
Detects all persons in a frame and returns bounding boxes with confidence scores.
"""

from ultralytics import YOLO
import numpy as np


class PersonDetector:
    def __init__(self, model_size: str = "n", confidence: float = 0.5):
        """
        Args:
            model_size: 'n' (nano/fast), 's' (small), 'm' (medium), 'l' (large)
            confidence: Minimum detection confidence threshold
        """
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.confidence = confidence
        print(f"[Detector] YOLOv8{model_size} loaded.")

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect all persons in frame.

        Returns:
            List of tuples: (x1, y1, x2, y2, confidence)
        """
        results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
        boxes = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Clamp to frame boundaries
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) > 20 and (y2 - y1) > 40:  # Skip tiny boxes
                        boxes.append((x1, y1, x2, y2, conf))

        return boxes
