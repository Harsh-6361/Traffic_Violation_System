# Traffic Violation Detection System

An AI-powered traffic violation detection system that uses **YOLOv8** for real-time helmet violation detection from dashcam footage and **PaddleOCR** for automatic license plate recognition.

## Features

- **Helmet Violation Detection** — Detects riders without helmets using a custom-trained YOLOv8 model.
- **Object Tracking** — Uses YOLO's built-in tracker to assign persistent IDs and avoid duplicate violations.
- **License Plate Recognition** — Reads vehicle number plates with PaddleOCR for automated evidence collection.
- **Evidence Logging** — Saves annotated frame images and a CSV report of all detected violations.

## Project Structure

```
Traffic_Violation_System/
├── main.py               # Main pipeline: detection, tracking, OCR, and reporting
├── src/
│   ├── detector.py       # TrafficDetector class wrapping the YOLO model
│   └── ocr_utils.py      # PlateReader class wrapping PaddleOCR
├── models/
│   └── best.pt           # Custom-trained YOLOv8 weights (helmet detection)
├── input_video/          # Input dashcam footage
├── output/
│   ├── frames/           # Saved violation evidence images
│   └── violation_log.csv # CSV report of detected violations
├── check_classes.py      # Utility to print model class names
├── check_video.py        # Utility to verify video file integrity
├── debug_view.py         # Utility to visualize detections in real time
├── requirements.txt      # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- A custom-trained YOLOv8 model (`models/best.pt`) with the following classes:
  - `0` — With helmet
  - `1` — Without helmet

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Harsh-6361/Traffic_Violation_System.git
   cd Traffic_Violation_System
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the main detection pipeline

Place your dashcam video at `input_video/dashcam_footage.mp4`, then run:

```bash
python main.py
```

The system will:
1. Process every 3rd frame of the video for efficiency.
2. Track detected objects across frames to assign unique IDs.
3. Log each unique "No Helmet" violation with a timestamp and detected license plate.
4. Save annotated evidence images to `output/frames/`.
5. Write a summary CSV report to `output/violation_log.csv`.

### Utility scripts

- **Check model classes** — Verify the class names in your trained model:

  ```bash
  python check_classes.py
  ```

- **Check video integrity** — Confirm the input video can be opened and read:

  ```bash
  python check_video.py
  ```

- **Debug view** — Visualize all detections in a live video window (press `q` to quit):

  ```bash
  python debug_view.py
  ```

## Output

| Column | Description |
|---|---|
| Track ID | Unique object tracker ID for the violator |
| Timestamp | Time position in the video (HH:MM:SS) |
| Violation | Type of violation detected |
| Vehicle Number (Detected) | License plate text from OCR |
| Evidence | Path to the saved annotated frame image |

## Dependencies

| Package | Purpose |
|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8 object detection and tracking |
| [opencv-python](https://github.com/opencv/opencv-python) | Video processing and image manipulation |
| [pandas](https://github.com/pandas-dev/pandas) | CSV report generation |
| [paddlepaddle](https://github.com/PaddlePaddle/Paddle) | Deep learning framework for OCR |
| [paddleocr](https://github.com/PaddlePaddle/PaddleOCR) | Optical character recognition for license plates |
| [numpy](https://github.com/numpy/numpy) | Numerical operations |

## License

This project is for educational and research purposes.
