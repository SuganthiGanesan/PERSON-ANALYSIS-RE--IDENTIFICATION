"""
body_analyzer.py
----------------
Estimates body type/build and approximate height ratio
using MediaPipe Pose landmark detection.
"""

import cv2
import numpy as np
import mediapipe as mp


class BodyAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[BodyAnalyzer] MediaPipe Pose loaded.")

    def analyze(self, person_crop: np.ndarray, box: tuple, frame_height: int) -> dict:
        """
        Analyze body build and height for a cropped person.

        Args:
            person_crop: Cropped BGR image of the person
            box: (x1, y1, x2, y2) bounding box in original frame
            frame_height: Height of the original frame

        Returns:
            dict with 'build' and 'height_estimate' keys
        """
        result = {
            "build": "Unknown",
            "height_estimate": self._estimate_height(box, frame_height),
            "shoulder_hip_ratio": None,
        }

        if person_crop is None or person_crop.size == 0:
            return result

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(rgb)

        if not pose_result.pose_landmarks:
            return result

        lm = pose_result.pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        try:
            # Get shoulder and hip landmark x positions
            l_shoulder = lm[PL.LEFT_SHOULDER].x
            r_shoulder = lm[PL.RIGHT_SHOULDER].x
            l_hip = lm[PL.LEFT_HIP].x
            r_hip = lm[PL.RIGHT_HIP].x

            shoulder_w = abs(l_shoulder - r_shoulder)
            hip_w = abs(l_hip - r_hip)
            ratio = shoulder_w / (hip_w + 1e-5)

            result["shoulder_hip_ratio"] = round(ratio, 2)

            if ratio >= 1.25:
                result["build"] = "Athletic / V-Shape"
            elif ratio <= 0.85:
                result["build"] = "Pear Shape"
            elif 0.95 <= ratio <= 1.1:
                result["build"] = "Hourglass"
            else:
                result["build"] = "Rectangle / Average"

        except Exception:
            pass

        return result

    def _estimate_height(self, box: tuple, frame_height: int) -> str:
        """Estimate relative height based on bounding box size."""
        _, y1, _, y2 = box[:4]
        ratio = (y2 - y1) / max(frame_height, 1)

        if ratio > 0.60:
            return "Tall (close to camera)"
        elif ratio > 0.40:
            return "Medium height"
        elif ratio > 0.20:
            return "Short / Average"
        else:
            return "Far from camera"

    def __del__(self):
        try:
            self.pose.close()
        except Exception:
            pass
