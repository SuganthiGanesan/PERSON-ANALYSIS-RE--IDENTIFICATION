"""
utils/draw.py
-------------
Drawing utilities for overlaying detection results on video frames.
Labels are drawn OUTSIDE the bounding box (below it, or above as a
fallback) so they never get squeezed onto the person's face/body when
the box fills most of the frame (e.g. when standing close to camera).
"""

import cv2
import numpy as np


# Palette: person IDs cycle through these BGR colours
ID_COLOURS = [
    (0, 200, 255),   # Cyan
    (0, 255, 100),   # Green
    (255, 100, 0),   # Blue
    (255, 0, 200),   # Magenta
    (0, 165, 255),   # Orange
    (147, 20, 255),  # Purple
    (0, 255, 255),   # Yellow
    (200, 50, 50),   # Dark Blue
    (0, 180, 255),   # Sky
    (180, 105, 255), # Pink
]


def get_id_colour(person_id: int) -> tuple:
    return ID_COLOURS[(person_id - 1) % len(ID_COLOURS)]


def draw_person(
    frame: np.ndarray,
    box: tuple,
    person_id: int,
    is_reid: bool,
    colour_name: str,
    build: str,
    height: str,
    confidence: float,
) -> np.ndarray:
    """
    Draw bounding box + info labels.
    Labels are drawn OUTSIDE the box (below, or above as fallback) so they
    never get squeezed onto the person's face when the box fills the frame.
    """
    x1, y1, x2, y2 = box[:4]
    fh, fw = frame.shape[:2]

    id_colour = get_id_colour(person_id)
    thickness = 3 if is_reid else 2

    # --- Bounding Box ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), id_colour, thickness)

    status_text = "RE-ID" if is_reid else "NEW"
    lines = [
        (f"#{person_id} [{status_text}]  Conf: {confidence:.0%}", id_colour),
        (f"Dress : {colour_name}", (220, 220, 220)),
        (f"Build : {build}",       (200, 200, 200)),
        (f"Height: {height}",      (180, 180, 180)),
    ]

    line_h, pad = 24, 8
    panel_h = len(lines) * line_h + pad * 2
    panel_w = max(240, min(x2 - x1, fw))  # never narrower than 240px, never wider than frame

    # Try placing panel just BELOW the box
    panel_y1 = y2 + 6
    panel_y2 = panel_y1 + panel_h
    if panel_y2 > fh:
        # No room below -> try ABOVE the box
        panel_y1 = y1 - panel_h - 6
        panel_y2 = y1 - 6
        if panel_y1 < 0:
            # Box fills the whole frame -> clamp to bottom edge as last resort
            panel_y2 = fh - 4
            panel_y1 = panel_y2 - panel_h

    panel_x1 = max(0, min(x1, fw - panel_w))
    panel_x2 = panel_x1 + panel_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (txt, col) in enumerate(lines):
        ty = panel_y1 + pad + (i + 1) * line_h - 6
        cv2.putText(frame, txt, (panel_x1 + 8, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    return frame


def draw_dashboard(frame: np.ndarray, stats: dict, fps: float) -> np.ndarray:
    """Top-left HUD with FPS and person count."""
    lines = [
        (f"FPS            : {fps:.1f}",                          (100, 255, 100)),
        (f"Unique Persons : {stats.get('total_unique_persons', 0)}", (255, 220, 100)),
        (f"[Q] Quit   [S] Screenshot",                           (160, 160, 160)),
    ]

    pad      = 8
    line_h   = 22
    panel_w  = 270
    panel_h  = len(lines) * line_h + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # thin coloured top border
    cv2.rectangle(frame, (0, 0), (panel_w, 3), (0, 200, 255), -1)

    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (pad, pad + (i + 1) * line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    return frame


def draw_no_detection(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    txt = "No person detected — point camera at a person"
    tw, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], None
    cv2.putText(frame, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    return frame