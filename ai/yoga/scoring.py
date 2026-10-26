"""Overall pose score: a documented weighted combination of three explainable parts.

    symmetric poses (T Pose, Chair Pose):   0.65 alignment + 0.20 stability + 0.15 symmetry
    one-sided poses (Tree, Warrior II, Triangle): 0.75 alignment + 0.25 stability

Symmetry is still *reported* for one-sided poses but is not scored, because those poses are
asymmetric by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

W_SYM = {"alignment": 0.65, "stability": 0.20, "symmetry": 0.15}
W_ASYM = {"alignment": 0.75, "stability": 0.25, "symmetry": 0.0}


@dataclass
class ScoreBreakdown:
    alignment: float
    stability: float
    symmetry: float
    overall: float
    grade: str


def grade_for(score: float) -> str:
    if not math.isfinite(score):
        return "No data"
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Needs work"


def compute_overall(alignment: float, stability: float, symmetry: float, symmetric_pose: bool) -> ScoreBreakdown:
    """Combine the three sub-scores (missing parts are dropped and weights re-normalised)."""
    weights = W_SYM if symmetric_pose else W_ASYM
    parts = {"alignment": alignment, "stability": stability, "symmetry": symmetry}
    acc, wsum = 0.0, 0.0
    for key, value in parts.items():
        w = weights[key]
        if w > 0 and value is not None and math.isfinite(value):
            acc += w * value
            wsum += w
    overall = acc / wsum if wsum > 0 else float("nan")
    return ScoreBreakdown(alignment, stability, symmetry, overall, grade_for(overall))
