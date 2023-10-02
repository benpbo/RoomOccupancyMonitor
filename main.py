from dataclasses import dataclass
import os
from collections import defaultdict
from datetime import datetime
import logging
import sys
from typing import Iterable, Callable, Sequence
import csv
import re

from argparse import ArgumentParser
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

# Model
MINIMUM_CONFIDENCE = 0.5

# Video prediction
# The smaller this value, the harder it is for the smoothed value to change
SMOOTH_FACTOR = 0.05
VIDEO_TEXT_POSITION = (7, 70)
VIDEO_TEXT_FONT_SCALE = 3
VIDEO_TEXT_COLOR = (100, 0, 255)
VIDEO_TEXT_THICKNESS = 3

# Data tag testing
VIDEO_FILE_NAME_REGEX = re.compile(
    r'IP Camera(\d)_NVR-OR(\d+)_NVR-OR\2_(\d{14})_(\d{14})_\d{7}_anonymized\.mp4')


@dataclass
class VideoArguments:
    video_path: str


@dataclass
class TagsArguments:
    tags_path: str


@dataclass
class DataTag:
    room: int
    timestamp: datetime
    count: int


@dataclass
class CameraVideoFile:
    file_name: str
    camera: int
    room: int
    start: datetime
    end: datetime


class CameraVideoFileRepository:
    def __init__(self):
        files = (parse_camera_video_file(file)
                 for file in os.listdir())
        files = [file for file in files
                 if file is not None]
        rooms = {file.room for file in files}
        self._files = defaultdict(lambda: [],
                                  {room: [file for file in files
                                          if file.room == room]
                                   for room in rooms})

    def get_file(self, room: int, time: datetime) -> Sequence[CameraVideoFile]:
        return [file
                for file in self._files[room]
                if file.start <= time <= file.end]


def smooth_value(new, previous) -> float:
    return SMOOTH_FACTOR * new + (1 - SMOOTH_FACTOR) * previous


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


def read_tags_from_csv(path: str) -> Iterable[DataTag]:
    with open(path) as tags_file:
        reader = csv.DictReader(tags_file)
        for entry in reader:
            day, month, year, hour, minute, second, room, count = map(
                int,
                (*entry['Date'].split('.'),
                 *entry['Time'].split('.'),
                 entry['Room'],
                 entry['Count']))
            timestamp = datetime(
                year, month, day,
                hour, minute, second)
            yield DataTag(room, timestamp, count)


def parse_camera_video_file(file_name: str) -> CameraVideoFile | None:
    match = VIDEO_FILE_NAME_REGEX.fullmatch(file_name)
    if match is None:
        return None

    camera, room, start, end = match.groups()
    camera, room = map(int, (camera, room))
    start, end = (datetime.strptime(date, '%Y%m%d%H%M%S')
                  for date in (start, end))
    return CameraVideoFile(file_name, camera, room, start, end)


def main(arguments: VideoArguments | TagsArguments):
    # Load the model
    logging.info('Loading model')
    model = YOLO('yolov8n.pt')

    match arguments:
        case VideoArguments(path):
            logging.info('Starting capture')
            capture = cv2.VideoCapture(path)
            results = predict_capture(model, capture)
            display_results(results)
            capture.release()
        case TagsArguments(path):
            repository = CameraVideoFileRepository()
            tags = read_tags_from_csv(path)
            test_data_tags(model, tags, repository)


if __name__ == '__main__':
    # Parse commandline arguments
    parser = ArgumentParser()
    argument_group = parser.add_mutually_exclusive_group()
    argument_group.add_argument(
        '-v', '--video',
        help='Path to video file that the model will be run on')
    argument_group.add_argument(
        '-t', '--tags',
        help='Path to tags CSV file that will be tested')

    arguments = parser.parse_args()
    video_path = arguments.video
    tags_path = arguments.tags

    if not video_path and not tags_path:
        parser.error('At least one of --video and --tags required')

    # Configure logging
    logging.config.fileConfig('logging.config')

    match (video_path, tags_path):
        case (video_path, None):
            main(VideoArguments(video_path))
        case (None, tags_path):
            main(TagsArguments(tags_path))
