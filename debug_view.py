import cv2
from ultralytics import YOLO

# CONFIG
VIDEO_PATH = "input_video/dashcam_footage.mp4"
MODEL_PATH = "models/best.pt"

def debug():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    print("Press 'q' to quit the video window.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection with VERY LOW confidence to see everything
        results = model(frame, conf=0.1) 
        
        # Plot the results directly on the frame
        annotated_frame = results[0].plot()

        # Show the video
        cv2.imshow("YOLO Debug View", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug()