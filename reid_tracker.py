"""
reid_tracker.py
---------------
Person Re-Identification system.

Each person is assigned a unique numeric ID.
When a person leaves and re-enters the frame, the system attempts to
match them to a previously seen person using appearance features
(colour histograms) + cosine similarity.
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple


@dataclass
class TrackedPerson:
    person_id: int
    features: List = field(default_factory=list)   # List of feature vectors
    colour: str = "Unknown"
    build: str = "Unknown"
    height: str = "Unknown"
    first_seen: str = ""
    last_seen: str = ""
    seen_count: int = 0
    is_active: bool = True


class PersonReID:
    def __init__(self, similarity_threshold: float = 0.72, max_stored_features: int = 15):
        """
        Args:
            similarity_threshold: Cosine similarity score above which a person is re-identified.
                                  Lower = more aggressive matching, Higher = stricter.
            max_stored_features: Max feature vectors kept per person (rolling window).
        """
        self.persons: Dict[int, TrackedPerson] = {}
        self.next_id = 1
        self.threshold = similarity_threshold
        self.max_features = max_stored_features

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract appearance feature vector from a person crop.
        Uses multi-region HSV colour histograms for robustness.
        """
        resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        features = []

        # Split into upper body and lower body and extract histograms separately
        regions = [
            hsv[:64, :],   # Upper body (torso)
            hsv[64:, :],   # Lower body (legs)
        ]

        for region in regions:
            for channel in range(3):
                hist = cv2.calcHist([region], [channel], None, [32], [0, 256])
                features.extend(hist.flatten())

        feature_vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(feature_vec)
        if norm > 0:
            feature_vec /= norm

        return feature_vec

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _compute_match_score(self, features: np.ndarray, person: TrackedPerson, colour: str) -> float:
        """
        Compute match score against a tracked person.
        Combines appearance similarity + colour agreement.
        """
        if not person.features:
            return 0.0

        avg_feature = np.mean(person.features, axis=0)
        avg_feature /= (np.linalg.norm(avg_feature) + 1e-8)

        appearance_score = float(cosine_similarity([features], [avg_feature])[0][0])

        # Small boost if colour matches
        colour_bonus = 0.05 if (colour == person.colour and colour != "Unknown") else 0.0

        return appearance_score + colour_bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, crop: np.ndarray, colour: str, build: str, height: str) -> Tuple[int, bool]:
        """
        Given a person crop and their attributes, find or create a person record.

        Returns:
            (person_id, is_reidentified)
            is_reidentified=True means this person was seen before.
        """
        if crop is None or crop.size == 0:
            return -1, False

        features = self._extract_features(crop)
        now = datetime.now().strftime("%H:%M:%S")

        best_id = None
        best_score = 0.0

        for pid, person in self.persons.items():
            score = self._compute_match_score(features, person, colour)
            if score > best_score:
                best_score = score
                best_id = pid

        if best_score >= self.threshold and best_id is not None:
            # Re-identified existing person
            p = self.persons[best_id]
            p.features.append(features)
            p.features = p.features[-self.max_features:]  # Rolling window
            p.last_seen = now
            p.seen_count += 1
            p.is_active = True
            return best_id, True
        else:
            # New person
            new_id = self.next_id
            self.next_id += 1
            self.persons[new_id] = TrackedPerson(
                person_id=new_id,
                features=[features],
                colour=colour,
                build=build,
                height=height,
                first_seen=now,
                last_seen=now,
                seen_count=1,
            )
            return new_id, False

    def get_person_info(self, person_id: int) -> Optional[TrackedPerson]:
        return self.persons.get(person_id)

    def get_all_persons(self) -> Dict:
        return self.persons

    def get_stats(self) -> Dict:
        total = len(self.persons)
        return {
            "total_unique_persons": total,
            "next_id": self.next_id,
        }
