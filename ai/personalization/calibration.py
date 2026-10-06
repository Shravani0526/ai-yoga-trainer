"""Personalised calibration.

The user holds a pose as well as they *comfortably* can for a few seconds. We store the
median of each measured metric. Calibration then adapts the target ranges in two bounded,
explainable ways:

1. Flexibility / mobility: if the user's comfortable value lies outside the canonical range,
   the violated side of the range is extended towards it, but never by more than
   ``MAX_SHIFT_ANGLE`` degrees (``MAX_SHIFT_DIST`` torso lengths for distances). Ranges are only
   ever widened, never tightened, and never shifted past what the user demonstrated.
2. Body proportions: stance width / foot-lift targets are scaled with the user's leg length
   relative to their torso (relative to a reference ratio of 2.0).

Calibration should be done with a good-faith attempt at the pose (ideally supervised): it
personalises the *limits*, it does not certify that a pose is safe for the user's body.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from ai.pose.geometry import compute_proportions
from ai.yoga.criteria import Criterion

MAX_SHIFT_ANGLE = 12.0
MAX_SHIFT_DIST = 0.30
REFERENCE_LEG_RATIO = 2.0
LEG_SCALED_METRICS = {"ankle_gap", "left_foot_lift", "right_foot_lift"}
MIN_SAMPLES_RECOMMENDED = 20


class CalibrationRecorder:
    """Collects metrics while the user holds their comfortable best version of a pose."""

    def __init__(self, pose: str, variant: str):
        self.pose = pose
        self.variant = variant
        self._metrics: List[Dict[str, float]] = []
        self._props: List[Dict[str, float]] = []

    def reset(self) -> None:
        self._metrics.clear()
        self._props.clear()

    @property
    def n_samples(self) -> int:
        return len(self._metrics)

    def add(self, metrics: Dict[str, float], coords: Optional[np.ndarray] = None) -> None:
        self._metrics.append(dict(metrics))
        if coords is not None:
            self._props.append(compute_proportions(coords))

    def build_profile(self) -> Dict:
        if not self._metrics:
            raise ValueError("No frames were recorded - make sure your whole body is visible.")
        names = sorted({k for m in self._metrics for k in m})
        stats: Dict[str, Dict[str, float]] = {}
        for name in names:
            vals = np.array([m.get(name, np.nan) for m in self._metrics], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            stats[name] = {"median": med, "mad": mad, "n": int(len(vals))}
        proportions: Dict[str, float] = {}
        for key in ("leg_ratio", "arm_ratio"):
            vals = np.array([p.get(key, np.nan) for p in self._props], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                proportions[key] = float(np.median(vals))
        return {
            "version": 1,
            "pose": self.pose,
            "variant": self.variant,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_samples": self.n_samples,
            "metrics": stats,
            "proportions": proportions,
        }


def _leg_scale(profile: Dict) -> float:
    ratio = profile.get("proportions", {}).get("leg_ratio")
    if ratio is None or not math.isfinite(ratio) or ratio <= 0:
        return 1.0
    return float(np.clip(ratio / REFERENCE_LEG_RATIO, 0.8, 1.25))


def apply_calibration(criteria: Sequence[Criterion], profile: Optional[Dict]) -> List[Criterion]:
    """Return personalised criteria (the input is returned unchanged if there is no profile)."""
    if not profile:
        return list(criteria)
    stats = profile.get("metrics", {})
    scale = _leg_scale(profile)
    out: List[Criterion] = []
    for c in criteria:
        low, high = c.low, c.high
        if c.metric in LEG_SCALED_METRICS and not c.is_angle:
            low, high = low * scale, high * scale
        s = stats.get(c.metric)
        if s is not None and math.isfinite(s.get("median", float("nan"))):
            m = s["median"]
            cap = MAX_SHIFT_ANGLE if c.is_angle else MAX_SHIFT_DIST
            if m > high:
                high = high + min(m - high, cap)
            elif m < low:
                low = low - min(low - m, cap)
        if c.is_angle:
            low, high = max(0.0, low), min(180.0, high)
        out.append(c.with_range(low, high))
    return out


def describe_adaptation(criteria: Sequence[Criterion], personalised: Sequence[Criterion]) -> List[Dict]:
    """Rows describing which ranges were changed by calibration (for the UI)."""
    rows = []
    for a, b in zip(criteria, personalised):
        if abs(a.low - b.low) > 1e-6 or abs(a.high - b.high) > 1e-6:
            rows.append({
                "Joint": a.label,
                "Default target": f"{a.low:.2f}–{a.high:.2f}" if not a.is_angle else f"{a.low:.0f}–{a.high:.0f}°",
                "Personal target": f"{b.low:.2f}–{b.high:.2f}" if not b.is_angle else f"{b.low:.0f}–{b.high:.0f}°",
            })
    return rows
