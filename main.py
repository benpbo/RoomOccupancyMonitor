import sys

import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

MINIMUM_CONFIDENCE = 0.5

# Load the model
model = YOLO('yolov8n.pt')

# Predict with the model
_, video_path = sys.argv
capture = cv2.VideoCapture(video_path)
while capture.isOpened():
    success, frame = capture.read()
    if success:
        results: list[Results] = model.predict(
            frame, conf=MINIMUM_CONFIDENCE, classes=0)
        result = results[0]

        detection_count = len(result.boxes.data)
        print(f'Detected {detection_count} persons')
    
        # Display the annotated frame
        annotated_frame = result.plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    
capture.release()
cv2.destroyAllWindows()
