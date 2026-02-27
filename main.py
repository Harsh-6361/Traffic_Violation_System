import cv2
import pandas as pd
import os
from datetime import timedelta
from src.detector import TrafficDetector
from src.ocr_utils import PlateReader
from ultralytics import YOLO

# CONFIGURATION
VIDEO_PATH = "input_video/dashcam_footage.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_CSV = "output/violation_log.csv"
OUTPUT_FRAMES = "output/frames/"

# Ensure output directories exist
os.makedirs(OUTPUT_FRAMES, exist_ok=True)

def main():
    # 1. Load Model directly here to access the 'track' feature
    model = YOLO(MODEL_PATH)
    
    # Initialize OCR
    reader = PlateReader()
    
    # Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    # Store processed IDs to prevent duplicates
    # Format: {track_id: timestamp}
    processed_ids = set()
    log_data = []

    print("Starting Smart Tracking Analysis...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # SKIP FRAMES: Process every 3rd frame (Tracking needs frequent frames to work)
        if frame_count % 3 != 0:
            continue

        # 2. RUN TRACKING
        # persist=True tells YOLO to keep remembering IDs between frames
        # conf=0.5 ensures we only look at high-confidence detections
        results = model.track(frame, persist=True, verbose=False, conf=0.5)
        
        # If nothing detected, skip
        if results[0].boxes.id is None:
            continue

        boxes = results[0].boxes
        
        # Get the IDs and Class IDs
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().tolist()

        # 3. Analyze each detected object
        for box, track_id, class_id in zip(xyxys, track_ids, class_ids):
            
            # Check if this object is a "No Helmet" violation
            # (Assuming Class 1 is 'Without helmet' based on your previous check)
            if class_id == 1: 
                
                # DUPLICATE CHECK: Has this ID been logged already?
                if track_id in processed_ids:
                    continue  # Skip it, we already caught him!

                # --- NEW VIOLATION DETECTED ---
                
                # 1. Mark as processed immediately
                processed_ids.add(track_id)
                
                # 2. Get Timestamp
                seconds = frame_count / fps
                timestamp = str(timedelta(seconds=seconds)).split(".")[0]
                
                # 3. SMART OCR (Guessing the Plate Location)
                # Logic: Plate is usually below the rider's head/body.
                # We crop a region below the detection box.
                x1, y1, x2, y2 = map(int, box)
                
                # Define search area for plate: 
                # Look starting from the bottom of the "No Helmet" box
                # Width: same as rider, Height: 200px downwards
                h, w, _ = frame.shape
                plate_y1 = min(y2, h)
                plate_y2 = min(y2 + 200, h) # Look 200px down
                plate_x1 = max(0, x1 - 50)  # Widen search slightly
                plate_x2 = min(w, x2 + 50)
                
                # Crop and Read
                plate_crop = [plate_x1, plate_y1, plate_x2, plate_y2]
                plate_text = reader.read_plate(frame, plate_crop)
                
                # Filter bad OCR results
                if len(plate_text) < 4:
                    plate_text = "Not Visible"

                print(f"[NEW VIOLATION] ID: {track_id} | Time: {timestamp} | Plate: {plate_text}")

                # 4. Save Image Evidence
                img_name = f"violation_ID{track_id}_{frame_count}.jpg"
                img_path = os.path.join(OUTPUT_FRAMES, img_name)
                
                # Draw boxes for the report image
                # Red box on Rider
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"ID: {track_id} NO HELMET", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                
                # Blue box on where we looked for the plate
                cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 0, 0), 2)
                
                cv2.imwrite(img_path, frame)

                # 5. Log Data
                log_data.append({
                    "Track ID": track_id,
                    "Timestamp": timestamp,
                    "Violation": "No Helmet",
                    "Vehicle Number (Detected)": plate_text,
                    "Evidence": img_path
                })

    cap.release()
    
    # Save Final Report
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Processing Complete. Unique Violations found: {len(log_data)}")
        print(f"Report saved to {OUTPUT_CSV}")
    else:
        print("No violations detected.")

if __name__ == "__main__":
    main()