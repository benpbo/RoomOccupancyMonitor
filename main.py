import logging
import sys
from typing import Iterable

from argparse import ArgumentParser
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

MINIMUM_CONFIDENCE = 0.5
# The smaller this value, the harder it is for the smoothed value to change
SMOOTH_FACTOR = 0.05
VIDEO_TEXT_POSITION = (7, 70)
VIDEO_TEXT_FONT_SCALE = 3
VIDEO_TEXT_COLOR = (100, 0, 255)
VIDEO_TEXT_THICKNESS = 3


def smooth_value(new, previous) -> float:
    return SMOOTH_FACTOR * new + (1 - SMOOTH_FACTOR) * previous


def predict_capture(model: Model, capture: cv2.VideoCapture) -> Iterable[Results]:
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        yield from model.predict(
            frame, conf=MINIMUM_CONFIDENCE, classes=0, verbose=False)


def put_detection_counter_text(image: cv2.UMat, detection_count: int):
    cv2.putText(
        image,
        str(detection_count),
        VIDEO_TEXT_POSITION,
        cv2.FONT_HERSHEY_SIMPLEX,
        VIDEO_TEXT_FONT_SCALE,
        VIDEO_TEXT_COLOR,
        thickness=VIDEO_TEXT_THICKNESS,
        lineType=cv2.LINE_AA)


class PersonCount:
    def __init__(self, initial_value: int):
        self._current_count = float(initial_value)

    def update(self, new: int):
        next_count = smooth_value(new, self._current_count)
        has_value_changed = round(next_count) != self.current
        self._current_count = next_count

        return has_value_changed

    @property
    def current(self) -> int:
        return round(self._current_count)


def display_results(results: Iterable[Results]):
    initial_result = next(results)
    person_count = PersonCount(len(initial_result.boxes.data))
    logging.info('Initial person count: %i', person_count.current)
    for result in results:
        # Count detections
        detection_count = len(result.boxes.data)
        logging.debug('Raw person detection count: %i', detection_count)
        if person_count.update(detection_count):
            logging.info('Person count changed: %i', person_count.current)

        # Display the annotated frame
        annotated_frame = result.plot()
        put_detection_counter_text(annotated_frame, person_count.current)
        cv2.imshow('YOLOv8 Inference', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main(video_path: str):
    # Load the model
    logging.info('Loading model')
    model = YOLO('yolov8n.pt')

    # Predict with the model
    logging.info('Starting capture')
    capture = cv2.VideoCapture(video_path)
    results = predict_capture(model, capture)
    display_results(results)

    capture.release()


if __name__ == '__main__':
    # Parse commandline arguments
    parser = ArgumentParser()
    parser.add_argument(
        'video',
        help='Path to video file that the model will be run on')

    arguments = parser.parse_args()
    video_path = arguments.video

    # Configure logging
    logging.config.fileConfig('logging.config')

    main(video_path)
