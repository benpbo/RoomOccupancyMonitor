import sys
from typing import Iterable

import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

MINIMUM_CONFIDENCE = 0.5
# The smaller this value, the harder it is for the smoothed value to change
SMOOTH_FACTOR = 0.05


def smooth_value(new, previous) -> float:
    return SMOOTH_FACTOR * new + (1 - SMOOTH_FACTOR) * previous


def predict_capture(model: Model, capture: cv2.VideoCapture) -> Iterable[Results]:
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        results: list[Results] = model.predict(
            frame, conf=MINIMUM_CONFIDENCE, classes=0)
        yield results[0]


def main(video_path: str):
    # Load the model
    model = YOLO('yolov8n.pt')

    # Predict with the model
    capture = cv2.VideoCapture(video_path)
    results = predict_capture(model, capture)
    smoothed_detection_count = len(next(results).boxes.data)  # Initial value
    for result in results:
        detection_count = len(result.boxes.data)
        smoothed_detection_count = smooth_value(
            smoothed_detection_count, detection_count)
        print(f'Detected {round(smoothed_detection_count)} persons')

        # Display the annotated frame
        annotated_frame = result.plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _, video_path = sys.argv
    main(video_path)
