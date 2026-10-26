"""Left/right symmetry of the body (reported for every pose, scored for symmetric poses)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

SYMMETRY_PAIRS = ("elbow", "shoulder", "hip", "knee")
MAX_ANGLE_DIFF = 30.0     # degrees of left/right difference that gives a 0 score
MAX_TILT = 15.0           # degrees of shoulder / hip line tilt that gives a 0 score


def symmetry_score(metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Return (score 0..100 or NaN, per-pair absolute angle difference in degrees)."""
    parts = []
    diffs: Dict[str, float] = {}
    for name in SYMMETRY_PAIRS:
        l, r = metrics.get(f"left_{name}", float("nan")), metrics.get(f"right_{name}", float("nan"))
        if math.isfinite(l) and math.isfinite(r):
            d = abs(l - r)
            diffs[name] = d
            parts.append(100.0 * (1.0 - min(d / MAX_ANGLE_DIFF, 1.0)))
    for tilt in ("shoulder_tilt", "hip_tilt"):
        v = metrics.get(tilt, float("nan"))
        if math.isfinite(v):
            diffs[tilt] = v
            parts.append(100.0 * (1.0 - min(v / MAX_TILT, 1.0)))
    if not parts:
        return float("nan"), diffs
    return float(sum(parts) / len(parts)), diffs
