import cv2
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_path):
        # Load the custom trained YOLO model
        self.model = YOLO(model_path)
        
        # UPDATED MAPPING based on your specific model
        # 0: With helmet, 1: Without helmet
        self.classes = {
            'helmet': 0,
            'no_helmet': 1
        }

    def detect(self, frame):
        """
        Runs inference on a single frame.
        """
        # We lower conf to 0.4 to catch more violations
        results = self.model(frame, verbose=False, conf=0.4)
        return results[0]

    def check_helmet_violation(self, boxes):
        """
        Logic: Simply look for Class ID 1 (Without helmet)
        """
        violations = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            
            # If the detected object is 'Without helmet' (ID 1)
            if cls_id == self.classes['no_helmet']:
                violations.append(box)
                
        return violations