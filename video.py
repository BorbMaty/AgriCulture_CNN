import torch
import cv2

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/lucy/Desktop/brassai_projekt/yolov5/runs/train/brassai_detector3/weights/best.pt')

# Set model to eval mode (optional but good practice)
model.eval()

# Capture video (0 = default webcam; or provide video file path)
cap = cv2.VideoCapture(0)  # replace 0 with 'your_video.mp4' for file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results.render()[0]

    cv2.imshow('YOLOv5 Detection', annotated_frame)

    # Exit if 'q' is pressed or window is closed
    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty('YOLOv5 Detection', cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
