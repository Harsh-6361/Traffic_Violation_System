from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging

# Suppress Paddle warnings to keep terminal clean
logging.getLogger("ppocr").setLevel(logging.ERROR)

class PlateReader:
    def __init__(self):
        # REMOVED 'show_log=False' to fix the error
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def read_plate(self, frame, bbox):
        """
        Crops the license plate from the frame and reads text.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop the license plate
        plate_img = frame[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            return "Unknown"

        # Preprocessing (Grayscale)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Run OCR
        try:
            result = self.ocr.ocr(gray, cls=True)
        except Exception:
            return "Error"
        
        # Extract text
        detected_text = ""
        if result and result[0]:
            for line in result[0]:
                detected_text += line[1][0] + " "
        
        return detected_text.strip() if detected_text else "Unreadable"