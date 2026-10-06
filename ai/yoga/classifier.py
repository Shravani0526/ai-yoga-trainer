"""Transparent, rule-based yoga pose classifier.

This is NOT a trained machine-learning model and no accuracy figure is claimed for it.
For every (pose, variant) the classifier scores only the *signature* rules of that pose
(its defining geometry) with a generous tolerance, then ranks the candidates. A pose is
reported only if its score passes ``MIN_CLASS_SCORE``; otherwise the frame is labelled
"No pose / transitional". Every number is visible in ``ai/yoga/poses.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai.yoga.criteria import Criterion, score_value
from ai.yoga.poses import POSES, get_pose

MIN_CLASS_SCORE = 55.0       # below this the frame is "No pose / transitional"
CLASS_TOL_MULTIPLIER = 1.5   # classification is more forgiving than alignment scoring
GATE_MIN_SCORE = 30.0        # a gate rule scoring below this rejects the pose outright
MIN_VISIBLE_WEIGHT = 0.5     # share of signature weight that must be measurable

NO_POSE = "No pose / transitional"


@dataclass
class ClassificationResult:
    pose: str                                   # best pose name or NO_POSE
    variant: Optional[str]
    score: float                                # 0..100 of the best candidate
    margin: float                               # best minus runner-up (other pose)
    ranking: List[Tuple[str, str, float]] = field(default_factory=list)  # (pose, variant, score)

    @property
    def is_pose(self) -> bool:
        return self.pose != NO_POSE

    def score_for(self, pose: str, variant: Optional[str] = None) -> float:
        best = 0.0
        for p, v, s in self.ranking:
            if p == pose and (variant is None or v == variant):
                best = max(best, s)
        return best


def score_candidate(metrics: Dict[str, float], criteria: Sequence[Criterion]) -> float:
    """Signature score (0..100) for one pose variant."""
    sig = [c for c in criteria if c.signature]
    if not sig:
        return 0.0
    total_w = sum(c.weight for c in sig)
    acc, vis = 0.0, 0.0
    for c in sig:
        value = metrics.get(c.metric, float("nan"))
        if value is None or not math.isfinite(value):
            continue
        s = score_value(value, c.low, c.high, c.tol * CLASS_TOL_MULTIPLIER)
        if c.gate and s < GATE_MIN_SCORE:
            return 0.0
        acc += c.weight * s
        vis += c.weight
    if vis / total_w < MIN_VISIBLE_WEIGHT:
        return 0.0
    return acc / vis


def classify(metrics: Dict[str, float], pose_names: Optional[Sequence[str]] = None) -> ClassificationResult:
    """Rank all poses (all variants) for the given metrics."""
    names = list(pose_names) if pose_names else list(POSES.keys())
    ranking: List[Tuple[str, str, float]] = []
    for name in names:
        spec = get_pose(name)
        for variant, criteria in spec.variants.items():
            ranking.append((name, variant, score_candidate(metrics, criteria)))
    ranking.sort(key=lambda r: r[2], reverse=True)

    best = ranking[0] if ranking else (NO_POSE, None, 0.0)
    runner_up = next((r for r in ranking if r[0] != best[0]), None)
    margin = best[2] - (runner_up[2] if runner_up else 0.0)
    if best[2] < MIN_CLASS_SCORE:
        return ClassificationResult(NO_POSE, None, best[2], margin, ranking)
    return ClassificationResult(best[0], best[1], best[2], margin, ranking)
