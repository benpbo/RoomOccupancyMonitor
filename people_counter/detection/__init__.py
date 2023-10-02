from typing import Iterable

import cv2
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

MINIMUM_CONFIDENCE = 0.5


def predict_frame(model: Model, frame: cv2.UMat) -> Iterable[Results]:
    return model.predict(
        frame, conf=MINIMUM_CONFIDENCE, classes=0, verbose=False)


def predict_capture(model: Model, capture: cv2.VideoCapture) -> Iterable[Results]:
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        yield from predict_frame(model, frame)


def predict_capture_at(model: Model, capture: cv2.VideoCapture, milliseconds: float) -> Iterable[Results]:
    capture.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
    success, frame = capture.read()
    if success:
        return predict_frame(model, frame)
