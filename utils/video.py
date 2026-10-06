"""Video / image input helpers (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file. Try an MP4 (H.264) file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(fps, count, w, h, count / fps if count else 0.0)


def iter_video_frames(path: str, target_fps: float = 10.0, max_seconds: float = 120.0,
                      max_side: int = 960) -> Iterator[Tuple[np.ndarray, float]]:
    """Yield ``(frame_bgr, timestamp_seconds)`` sampled at ~``target_fps``.

    Timestamps come from the video clock (frame index / fps), so the hold timer measures
    *video* time and is independent of how fast the server processes frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file. Try an MP4 (H.264) file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    step = max(1, int(round(fps / max(target_fps, 1.0))))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if t > max_seconds:
                break
            if idx % step == 0:
                yield resize_max_side(frame, max_side), t
            idx += 1
    finally:
        cap.release()


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    s = max_side / float(longest)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def decode_image(data: bytes, max_side: int = 960) -> Optional[np.ndarray]:
    """Decode JPEG/PNG bytes into a BGR image (None when decoding fails)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None if img is None else resize_max_side(img, max_side)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
