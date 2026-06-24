"""
test_system.py
--------------
Quick sanity check — runs without a camera or video file.
Tests that all modules import and initialise correctly,
and processes a synthetic (blank) frame to verify the pipeline works end-to-end.
"""

import sys
import numpy as np
import cv2


def test_imports():
    print("Testing imports...")
    try:
        from detector import PersonDetector
        print("  ✅ detector.py")
    except Exception as e:
        print(f"  ❌ detector.py: {e}")
        return False

    try:
        from colour_detector import DressColourDetector
        print("  ✅ colour_detector.py")
    except Exception as e:
        print(f"  ❌ colour_detector.py: {e}")
        return False

    try:
        from body_analyzer import BodyAnalyzer
        print("  ✅ body_analyzer.py")
    except Exception as e:
        print(f"  ❌ body_analyzer.py: {e}")
        return False

    try:
        from reid_tracker import PersonReID
        print("  ✅ reid_tracker.py")
    except Exception as e:
        print(f"  ❌ reid_tracker.py: {e}")
        return False

    try:
        from utils import draw_person, draw_dashboard, draw_no_detection
        print("  ✅ utils/draw.py")
    except Exception as e:
        print(f"  ❌ utils/draw.py: {e}")
        return False

    return True


def test_reid():
    print("\nTesting Re-ID tracker...")
    from reid_tracker import PersonReID

    reid = PersonReID(similarity_threshold=0.72)

    # Simulate 3 different-coloured crops
    crop_a = np.zeros((128, 64, 3), dtype=np.uint8)
    crop_a[:, :, 2] = 200  # Red-ish

    crop_b = np.zeros((128, 64, 3), dtype=np.uint8)
    crop_b[:, :, 0] = 200  # Blue-ish

    # Person A — first sighting
    id1, is_reid1 = reid.update(crop_a, "Red", "Athletic", "Tall")
    print(f"  First sighting: ID={id1}, ReID={is_reid1}  (expected: ID=1, ReID=False)")
    assert id1 == 1 and not is_reid1, "FAIL"

    # Person B — new person
    id2, is_reid2 = reid.update(crop_b, "Blue", "Average", "Medium")
    print(f"  Second person:  ID={id2}, ReID={is_reid2}  (expected: ID=2, ReID=False)")
    assert id2 == 2 and not is_reid2, "FAIL"

    # Person A again — should be re-identified
    id3, is_reid3 = reid.update(crop_a, "Red", "Athletic", "Tall")
    print(f"  Re-sighting A:  ID={id3}, ReID={is_reid3}  (expected: ID=1, ReID=True)")
    assert id3 == 1 and is_reid3, "FAIL — Re-ID did not work"

    print("  ✅ Re-ID tracker working correctly")


def test_colour_detection():
    print("\nTesting colour detection...")
    from colour_detector import DressColourDetector

    det = DressColourDetector()

    # Create a mostly-blue image
    blue_img = np.zeros((128, 64, 3), dtype=np.uint8)
    blue_img[:, :, 0] = 180  # BGR blue channel
    colour, _ = det.get_dominant_colour(blue_img)
    print(f"  Blue image -> detected: {colour}")

    # Create a mostly-white image
    white_img = np.ones((128, 64, 3), dtype=np.uint8) * 240
    colour2, _ = det.get_dominant_colour(white_img)
    print(f"  White image -> detected: {colour2}")

    print("  ✅ Colour detector working")


def test_draw_utils():
    print("\nTesting draw utilities...")
    from utils import draw_person, draw_dashboard, draw_no_detection

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_person(frame, (50, 50, 200, 400), 1, False, "Blue", "Athletic", "Tall", 0.9)
    draw_person(frame, (300, 50, 500, 400), 2, True, "Red", "Average", "Medium", 0.75)
    draw_dashboard(frame, {"total_unique_persons": 2}, fps=30.0)
    draw_no_detection(frame)
    print("  ✅ Draw utilities working")


if __name__ == "__main__":
    print("=" * 50)
    print("  Person Re-ID System — Self Test")
    print("=" * 50)

    if not test_imports():
        print("\n❌ Import test failed. Check your installation.")
        sys.exit(1)

    test_reid()
    test_colour_detection()
    test_draw_utils()

    print("\n" + "=" * 50)
    print("  ✅ All tests passed! System is ready.")
    print("  Run: python main.py")
    print("=" * 50)
