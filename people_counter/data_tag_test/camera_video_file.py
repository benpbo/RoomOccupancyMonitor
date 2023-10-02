from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
import re
import os

VIDEO_FILE_NAME_REGEX = re.compile(
    r'IP Camera(\d)_NVR-OR(\d+)_NVR-OR\2_(\d{14})_(\d{14})_\d{7}_anonymized\.mp4')


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


def parse_camera_video_file(file_name: str) -> CameraVideoFile | None:
    match = VIDEO_FILE_NAME_REGEX.fullmatch(file_name)
    if match is None:
        return None

    camera, room, start, end = match.groups()
    camera, room = map(int, (camera, room))
    start, end = (datetime.strptime(date, '%Y%m%d%H%M%S')
                  for date in (start, end))
    return CameraVideoFile(file_name, camera, room, start, end)
