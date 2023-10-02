import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Iterable
import logging

from argparse import ArgumentParser
import cv2
from ultralytics import YOLO

from .data_tag_test import DataTag, test_data_tags
from .data_tag_test.camera_video_file import CameraVideoFileRepository
from .detection import predict_capture
from .display import display_results


@dataclass
class VideoArguments:
    video_path: str


@dataclass
class TagsArguments:
    tags_path: str


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
