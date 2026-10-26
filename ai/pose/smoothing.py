"""Temporal smoothing of keypoints with a One-Euro filter.

The One-Euro filter (Casiez et al., CHI 2012) is a low-pass filter whose cut-off rises with
the signal speed: it removes jitter while a person holds still, but adds almost no lag when
they move. Missing keypoints are held for a short time and then dropped (never invented).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ai.pose.keypoints import NUM_KEYPOINTS


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class KeypointSmoother:
    """One-Euro filter applied to a (17, 2) coordinate array."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 1.5, d_cutoff: float = 1.0,
                 max_gap_s: float = 0.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_gap_s = float(max_gap_s)
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx = np.zeros((NUM_KEYPOINTS, 2))
        self._last_t: Optional[float] = None
        self._missing_s = np.zeros(NUM_KEYPOINTS)
        self._has = np.zeros(NUM_KEYPOINTS, dtype=bool)

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        if self._x is None:
            self._x = np.where(np.isnan(coords), 0.0, coords)
            self._has = ~np.isnan(coords).any(axis=1)
            self._last_t = t
            return coords.copy()

        dt = max(t - (self._last_t if self._last_t is not None else t), 1e-3)
        dt = min(dt, 0.5)
        self._last_t = t

        out = np.full_like(coords, np.nan)
        valid = ~np.isnan(coords).any(axis=1)

        for i in range(NUM_KEYPOINTS):
            if valid[i]:
                self._missing_s[i] = 0.0
                if not self._has[i]:
                    self._x[i] = coords[i]
                    self._dx[i] = 0.0
                    self._has[i] = True
                    out[i] = coords[i]
                    continue
                dx = (coords[i] - self._x[i]) / dt
                a_d = _alpha(np.full(2, self.d_cutoff), dt)
                self._dx[i] = a_d * dx + (1.0 - a_d) * self._dx[i]
                cutoff = self.min_cutoff + self.beta * np.abs(self._dx[i])
                a = _alpha(cutoff, dt)
                self._x[i] = a * coords[i] + (1.0 - a) * self._x[i]
                out[i] = self._x[i]
            else:
                self._missing_s[i] += dt
                if self._has[i] and self._missing_s[i] <= self.max_gap_s:
                    out[i] = self._x[i]          # hold the last estimate briefly
                else:
                    self._has[i] = False         # forget: next sighting restarts the filter
        return out
