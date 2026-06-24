# 🧠 Person Analysis & Re-Identification System
**AI & Machine Learning Intern — Task 1**

A real-time computer vision system that detects persons from a camera feed, analyses their appearance (dress colour, body build, height), assigns unique IDs, and **re-identifies them** when they re-appear in the frame.

---

## 📁 Folder Structure

```
person_reid/
├── main.py                  # ← Entry point — run this
├── detector.py              # YOLOv8 person detection
├── colour_detector.py       # HSV-based dress colour detection
├── body_analyzer.py         # MediaPipe body build + height estimation
├── reid_tracker.py          # Re-Identification using cosine similarity
├── utils/
│   ├── __init__.py
│   └── draw.py              # Bounding box + HUD drawing utilities
├── test_system.py           # Self-test (no camera needed)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> On first run, YOLOv8 will automatically download the model weights (~6MB for nano).

---

## 🚀 Running the System

### Webcam (default)
```bash
python main.py
```

### Video File
```bash
python main.py --source path/to/video.mp4
```

### Single Image
```bash
python main.py --source path/to/image.jpg --image-mode
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | `0` = webcam, or path to video/image |
| `--model` | `n` | YOLOv8 size: `n` (fast) → `s` → `m` → `l` (accurate) |
| `--confidence` | `0.5` | Detection confidence (0.0–1.0) |
| `--reid-threshold` | `0.72` | Re-ID similarity score (lower = more aggressive) |
| `--no-body-analysis` | off | Skip MediaPipe for faster performance |
| `--image-mode` | off | Process single image |

---

## 🧪 Test Without Camera
```bash
python test_system.py
```
Runs all module checks and pipeline tests using synthetic images.

---

## 🎯 How It Works

### Detection
- **YOLOv8** detects all persons in each frame (class 0)
- Returns bounding boxes with confidence scores

### Dress Colour Detection
- Crops the **torso region** (20%–65% height, avoiding face/legs)
- Converts to **HSV colour space** for lighting robustness
- Matches against 10 predefined colour ranges (Red, Blue, Green, etc.)

### Body Analysis (MediaPipe)
- Extracts **pose landmarks** from the person crop
- Computes **shoulder-to-hip width ratio** for body shape:
  - `>1.25` → Athletic / V-Shape
  - `~1.0` → Rectangle / Average
  - `0.85–0.95` → Hourglass
  - `<0.85` → Pear Shape
- Estimates **relative height** based on bounding box size vs frame

### Re-Identification
- Extracts **multi-region HSV colour histograms** from upper + lower body
- Normalises into a feature vector
- Computes **cosine similarity** against all known person profiles
- If similarity ≥ threshold → **Re-Identified** (green border, "RE-ID" badge)
- If below threshold → **New Person** (red border, "NEW" badge)
- Maintains a **rolling window** of the last 15 feature vectors per person

---

## ✅ Requirements Coverage

| Requirement | Status | How |
|-------------|--------|-----|
| Person Detection | ✅ | YOLOv8 |
| Dress Colour Detection | ✅ | HSV histogram |
| User Re-Identification | ✅ | Cosine similarity |
| Height Estimation (Bonus) | ✅ | Bounding box ratio |
| Body Shape Classification (Bonus) | ✅ | MediaPipe pose |

---

## 🖥️ Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save screenshot to `screenshots/` folder |

---

## 🔧 Tuning Tips

- **Too many false Re-IDs?** → Increase `--reid-threshold` (e.g. `0.80`)
- **Missing Re-IDs?** → Decrease `--reid-threshold` (e.g. `0.65`)
- **Low FPS?** → Use `--model n` and `--no-body-analysis`
- **Better accuracy?** → Use `--model s` or `--model m`

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **YOLOv8** (Ultralytics) — Person detection
- **OpenCV** — Video capture, image processing
- **MediaPipe** — Pose estimation
- **scikit-learn** — Cosine similarity
- **NumPy** — Feature vector operations

---

