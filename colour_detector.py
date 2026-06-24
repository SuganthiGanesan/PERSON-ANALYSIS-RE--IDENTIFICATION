"""
colour_detector.py
------------------
Detects dominant dress colour from a cropped person image.
Uses HSV colour space for robust detection under different lighting.
"""

import cv2
import numpy as np


from typing import Tuple


# HSV ranges for common colours
COLOUR_RANGES = {
    "Red":    [([0, 100, 70], [10, 255, 255]), ([170, 100, 70], [180, 255, 255])],
    "Orange": [([10, 100, 100], [25, 255, 255])],
    "Yellow": [([25, 100, 100], [35, 255, 255])],
    "Green":  [([36, 50, 70], [89, 255, 255])],
    "Blue":   [([90, 50, 70], [130, 255, 255])],
    "Purple": [([130, 50, 70], [160, 255, 255])],
    "Pink":   [([160, 30, 150], [170, 255, 255])],
    "White":  [([0, 0, 200], [180, 30, 255])],
    "Black":  [([0, 0, 0], [180, 255, 50])],
    "Gray":   [([0, 0, 51], [180, 30, 199])],
}

# BGR colours for drawing labels
LABEL_COLOURS = {
    "Red": (0, 0, 200),
    "Orange": (0, 140, 255),
    "Yellow": (0, 215, 255),
    "Green": (0, 180, 0),
    "Blue": (255, 100, 0),
    "Purple": (180, 0, 180),
    "Pink": (180, 105, 255),
    "White": (220, 220, 220),
    "Black": (80, 80, 80),
    "Gray": (150, 150, 150),
    "Unknown": (100, 100, 100),
}


class DressColourDetector:
    def __init__(self):
        pass

    def get_dominant_colour(self, person_crop: np.ndarray) -> tuple[str, tuple]:
        """
        Detect dominant dress colour from person crop.

        Returns:
            (colour_name, bgr_colour_for_drawing)
        """
        if person_crop is None or person_crop.size == 0:
            return "Unknown", LABEL_COLOURS["Unknown"]

        h, w = person_crop.shape[:2]

        # Focus on torso region (avoid face and legs for better colour detection)
        torso_y1 = int(h * 0.20)
        torso_y2 = int(h * 0.65)
        torso_x1 = int(w * 0.15)
        torso_x2 = int(w * 0.85)
        torso = person_crop[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso.size == 0:
            torso = person_crop

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        max_count = 0
        dominant = "Unknown"

        for colour_name, ranges in COLOUR_RANGES.items():
            count = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                count += cv2.countNonZero(mask)

            if count > max_count:
                max_count = count
                dominant = colour_name

        return dominant, LABEL_COLOURS.get(dominant, LABEL_COLOURS["Unknown"])
