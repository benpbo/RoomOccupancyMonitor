from typing import Iterable
import logging

import cv2
from ultralytics.engine.results import Results

from .detection.person_count import PersonCount


VIDEO_TEXT_POSITION = (7, 70)
VIDEO_TEXT_FONT_SCALE = 3
VIDEO_TEXT_COLOR = (100, 0, 255)
VIDEO_TEXT_THICKNESS = 3


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
