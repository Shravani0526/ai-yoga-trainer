"""Stability tracking: how still is the person over the last ~1.5 seconds?

Two cues are combined (both measured in torso lengths so they do not depend on distance):
    * joint motion  - std-dev of each body joint's *body-centred* position over the window
    * body sway     - std-dev of the mid-hip position in the image over the window
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ai.pose.keypoints import BODY_INDICES

MAX_MOTION = 0.08          # torso lengths of motion that gives stability 0
MIN_SAMPLES = 5


class StabilityTracker:
    def __init__(self, window_s: float = 1.5, max_motion: float = MAX_MOTION):
        self.window_s = float(window_s)
        self.max_motion = float(max_motion)
        self._buf: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque()
        self.last_score = float("nan")

    def reset(self) -> None:
        self._buf.clear()
        self.last_score = float("nan")

    def update(self, t: float, coords: np.ndarray, center_in_torso: np.ndarray) -> float:
        """Add a frame and return the current stability score (0..100)."""
        self._buf.append((t, np.asarray(coords)[BODY_INDICES].copy(), np.asarray(center_in_torso, dtype=float)))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        if len(self._buf) < MIN_SAMPLES:
            return self.last_score if math.isfinite(self.last_score) else 100.0

        joints = np.stack([b[1] for b in self._buf])        # (n, 12, 2)
        centers = np.stack([b[2] for b in self._buf])       # (n, 2)
        with np.errstate(all="ignore"):
            joint_std = np.nanstd(joints, axis=0)            # (12, 2)
            joint_motion = float(np.nanmean(joint_std)) if np.isfinite(joint_std).any() else 0.0
            sway = float(np.sqrt(np.nanvar(centers[:, 0]) + np.nanvar(centers[:, 1])))
        if not math.isfinite(joint_motion):
            joint_motion = 0.0
        if not math.isfinite(sway):
            sway = 0.0
        motion = joint_motion + 0.5 * sway
        self.last_score = float(np.clip(100.0 * (1.0 - motion / self.max_motion), 0.0, 100.0))
        return self.last_score
