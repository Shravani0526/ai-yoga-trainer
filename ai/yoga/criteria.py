"""Rule definition shared by classification, alignment and corrections.

A :class:`Criterion` is one transparent, human-readable rule such as
"right knee angle must be 85-115 degrees". Nothing here is learned from data: every
number is an explicit, editable target range.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Iterable, List, Tuple

from ai.pose.keypoints import mirror_index

DEG = "°"


@dataclass(frozen=True)
class Criterion:
    key: str                 # unique id inside a pose variant, e.g. "right_knee"
    label: str               # text shown to the user, e.g. "Right knee"
    metric: str              # metric computed by ai.pose.geometry.compute_metrics
    low: float               # lower bound of the target range
    high: float              # upper bound of the target range
    tol: float = 30.0        # distance outside the range at which the score reaches 0
    weight: float = 1.0
    too_low: str = ""        # action when value < low   (imperative, no full stop)
    too_high: str = ""       # action when value > high
    unit: str = DEG          # DEG for angles, "x torso" for distances
    joints: Tuple[int, ...] = ()   # keypoint indices highlighted on the skeleton
    signature: bool = False  # used by the pose classifier
    gate: bool = False       # classifier rejects the pose if this rule is badly violated
    tip: str = ""            # practice drill used by the recommendation engine

    def with_range(self, low: float, high: float) -> "Criterion":
        return replace(self, low=float(low), high=float(high))

    @property
    def is_angle(self) -> bool:
        return self.unit == DEG


def deviation(value: float, low: float, high: float) -> float:
    """Signed distance to the target range: <0 below, >0 above, 0 inside."""
    if value < low:
        return value - low
    if value > high:
        return value - high
    return 0.0


def score_value(value: float, low: float, high: float, tol: float) -> float:
    """0..100 score: 100 inside the range, linear fall-off to 0 at ``tol`` outside it."""
    if value is None or not math.isfinite(value):
        return float("nan")
    dev = abs(deviation(value, low, high))
    if tol <= 0:
        return 100.0 if dev == 0 else 0.0
    return float(max(0.0, 100.0 * (1.0 - dev / tol)))


# --------------------------------------------------------------------------- mirroring
_SIDE_RE = re.compile(r"(?<![A-Za-z])(left|right)(?![A-Za-z])", re.IGNORECASE)


def _swap_side_word(match: "re.Match[str]") -> str:
    word = match.group(0)
    swapped = "right" if word.lower() == "left" else "left"
    if word.isupper():
        return swapped.upper()
    if word[0].isupper():
        return swapped.capitalize()
    return swapped


def swap_sides(text: str) -> str:
    """Swap left <-> right in a string (``upright`` is untouched)."""
    return _SIDE_RE.sub(_swap_side_word, text)


def mirror_criterion(c: Criterion) -> Criterion:
    """Return the same rule for the opposite body side."""
    return replace(
        c,
        key=swap_sides(c.key),
        label=swap_sides(c.label),
        metric=swap_sides(c.metric),
        too_low=swap_sides(c.too_low),
        too_high=swap_sides(c.too_high),
        tip=swap_sides(c.tip),
        joints=tuple(mirror_index(j) for j in c.joints),
    )


def mirror_criteria(criteria: Iterable[Criterion]) -> List[Criterion]:
    return [mirror_criterion(c) for c in criteria]
