from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CameraSource = int | str


@dataclass(frozen=True)
class CameraConfig:
    source: CameraSource = 0
    width: int | None = 1280
    height: int | None = 720


def parse_camera_source(value: str) -> CameraSource:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def open_camera(config: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.source)
    if config.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    if config.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera source: {config.source!r}")
    return cap


def read_rgb_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError("Could not read a frame from the camera.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def capture_frame(config: CameraConfig) -> np.ndarray:
    cap = open_camera(config)
    try:
        return read_rgb_frame(cap)
    finally:
        cap.release()


def scan_camera_indices(max_index: int = 5) -> list[int]:
    found: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
        finally:
            cap.release()
    return found


def load_rgb_image(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not load image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
