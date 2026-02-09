import cv2
import os

# Define the path exactly as your main code uses it
video_path = "input_video/dashcam_footage.mp4"

print("--- DIAGNOSTIC START ---")

# 1. Check if file exists on disk
if os.path.exists(video_path):
    print(f"✅ File found at: {video_path}")
    print(f"   File size: {os.path.getsize(video_path) / 1024:.2f} KB")
else:
    print(f"❌ ERROR: File NOT found at: {video_path}")
    print(f"   Current working directory is: {os.getcwd()}")
    print("   Please check inside the 'input_video' folder.")
    exit()

# 2. Try to open with OpenCV
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: OpenCV failed to open the video. (Codec or Corrupt file?)")
else:
    print("✅ OpenCV successfully opened the video source.")
    
    # 3. Try to read one frame
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame Read Successfully. Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("❌ ERROR: opened the file, but could not read any frames (Video might be empty/corrupt).")

cap.release()
print("--- DIAGNOSTIC END ---")