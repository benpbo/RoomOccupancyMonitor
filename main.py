import sys
from typing import Iterable

import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

MINIMUM_CONFIDENCE = 0.5

def predict_capture(model: Model, capture: cv2.VideoCapture) -> Iterable[Results]:
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        results: list[Results] = model.predict(
            frame, conf=MINIMUM_CONFIDENCE, classes=0)
        yield results[0]

# Load the model
model = YOLO('yolov8n.pt')

# Predict with the model
_, video_path = sys.argv
capture = cv2.VideoCapture(video_path)
for result in predict_capture(model, capture):
    detection_count = len(result.boxes.data)
    print(f'Detected {detection_count} persons')

    # Display the annotated frame
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
capture.release()
cv2.destroyAllWindows()
