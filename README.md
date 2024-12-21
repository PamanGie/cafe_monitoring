# Cafe Service Time Analysis using YOLO and DeepSORT

This project analyzes customer service time in a cafe using computer vision. It combines YOLO object detection for identifying baristas and customers, with DeepSORT for tracking their movements and calculating service durations.

## Features

- Custom YOLO model for detecting baristas and customers
- DeepSORT tracking for consistent ID assignment
- Duration calculation for:
  - Customer waiting time
  - Barista working time
- Output generation:
  - Annotated video with tracking visualization
  - CSV data export with tracking statistics

## Requirements

- Python 3.8+
- ultralytics
- deep-sort-realtime
- opencv-python-headless
- numpy
- pandas
- tqdm

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cafe-monitoring.git
cd cafe-monitoring

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your input video in `data/input/cafe.mp4`
2. Place your YOLO model in `data/models/best.pt`
3. Run the tracking script:
```bash
python cafe_tracker.py
```

## Output

The system generates:

- **Video Output**: Annotated video with bounding boxes, tracking IDs, and duration information
- **CSV Data**: Tracking results including:
  - Track ID
  - Class (Barista/Customer)
  - Duration
  - Frame information

## Acknowledgments

- YOLO by Ultralytics
- DeepSORT Real-time Object Tracking
- Laura Angelia (https://www.youtube.com/watch?v=wz0pvSFSW2A&t=309s) - Kindly Subscribe Her Channel! 

- ## Note
- Please use your own YOLO Model (This AI Compatible to YOLOv8 to YOLO11 model)
- You should use your own Video to Track

# Result
![Tracking Result](https://i.postimg.cc/GhCjvcB4/vlcsnap-2024-12-21-19h03m34s586.png "Cafe Service Tracking")
