from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
import logging

import cv2
from ultralytics.engine.model import Model

from .camera_video_file import CameraVideoFileRepository
from ..detection import predict_capture_at


@dataclass
class DataTag:
    room: int
    timestamp: datetime
    count: int


def test_data_tags(model: Model, tags: Iterable[DataTag], repository: CameraVideoFileRepository):
    for tag in tags:
        video_files = repository.get_file(tag.room, tag.timestamp)
        for video_file in video_files:
            capture = cv2.VideoCapture(video_file.file_name)
            timestamp_in_capture = tag.timestamp - video_file.start
            capture_milliseconds = timestamp_in_capture.total_seconds() * 1000
            results = predict_capture_at(model, capture, capture_milliseconds)
            for result in results:
                person_count = len(result.boxes.data)
                logging.info(
                    f'[{tag.timestamp}] At room {tag.room} using camera {video_file.camera} expected {tag.count}, found {person_count}')
