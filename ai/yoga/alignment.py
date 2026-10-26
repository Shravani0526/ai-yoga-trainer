"""Alignment analysis: compare measured joint metrics with a pose's target ranges."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ai.yoga.criteria import Criterion, deviation, score_value

STATUS_OK = "ok"
STATUS_LOW = "low"
STATUS_HIGH = "high"
STATUS_MISSING = "missing"


@dataclass
class JointResult:
    key: str
    label: str
    metric: str
    value: float
    low: float
    high: float
    deviation: float          # signed distance to the target range (0 = inside)
    score: float              # 0..100 (NaN when the joint is not visible)
    status: str               # ok / low / high / missing
    weight: float
    unit: str
    tol: float
    joints: tuple
    too_low: str
    too_high: str

    @property
    def severity(self) -> float:
        """0 = fine, 1 = as bad as the tolerance allows (can exceed 1)."""
        if self.status in (STATUS_OK, STATUS_MISSING) or self.tol <= 0:
            return 0.0
        return abs(self.deviation) / self.tol


@dataclass
class AlignmentResult:
    score: float                      # weighted 0..100, NaN if nothing visible
    joint_results: List[JointResult] = field(default_factory=list)
    visible_weight_fraction: float = 0.0

    @property
    def by_key(self) -> Dict[str, JointResult]:
        return {j.key: j for j in self.joint_results}

    def failing(self) -> List[JointResult]:
        return [j for j in self.joint_results if j.status in (STATUS_LOW, STATUS_HIGH)]


def evaluate_alignment(metrics: Dict[str, float], criteria: Sequence[Criterion]) -> AlignmentResult:
    """Score every criterion and combine them into a weighted alignment score."""
    results: List[JointResult] = []
    total_w = sum(c.weight for c in criteria) or 1.0
    vis_w, acc = 0.0, 0.0
    for c in criteria:
        value = metrics.get(c.metric, float("nan"))
        if value is None or not math.isfinite(value):
            results.append(JointResult(c.key, c.label, c.metric, float("nan"), c.low, c.high,
                                       float("nan"), float("nan"), STATUS_MISSING, c.weight,
                                       c.unit, c.tol, c.joints, c.too_low, c.too_high))
            continue
        dev = deviation(value, c.low, c.high)
        sc = score_value(value, c.low, c.high, c.tol)
        status = STATUS_OK if dev == 0 else (STATUS_LOW if dev < 0 else STATUS_HIGH)
        results.append(JointResult(c.key, c.label, c.metric, float(value), c.low, c.high, float(dev),
                                   sc, status, c.weight, c.unit, c.tol, c.joints, c.too_low, c.too_high))
        vis_w += c.weight
        acc += c.weight * sc
    score = acc / vis_w if vis_w > 0 else float("nan")
    return AlignmentResult(score=score, joint_results=results, visible_weight_fraction=vis_w / total_w)
