"""Explainable joint-level corrections.

Every correction states four things: the CURRENT value, the TARGET range, the DEVIATION
from that range and a plain-language instruction, e.g.

    "Right knee: 132°. Target: 90–110°. Bend your right knee slightly."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from ai.yoga.alignment import JointResult, STATUS_HIGH, STATUS_LOW, STATUS_OK
from ai.yoga.criteria import DEG

# A joint is only reported when its own score is below this (avoids nagging over 1-2 degrees).
CORRECTION_SCORE_MAX = 85.0


@dataclass
class Correction:
    key: str
    label: str
    value: float
    low: float
    high: float
    deviation: float          # signed: <0 below the range, >0 above
    unit: str
    direction: str            # "low" or "high"
    severity: float           # deviation / tolerance
    action: str               # imperative instruction (without adverb)
    instruction: str          # the instruction sentence, e.g. "Bend your right knee slightly."
    text: str                 # full explainable sentence
    joints: tuple

    def as_dict(self) -> dict:
        return {
            "joint": self.label,
            "current": format_value(self.value, self.unit),
            "target": format_range(self.low, self.high, self.unit),
            "deviation": format_deviation(self.deviation, self.unit),
            "correction": self.instruction,
        }


def format_value(value: float, unit: str) -> str:
    if not math.isfinite(value):
        return "n/a"
    if unit == DEG:
        return f"{value:.0f}{DEG}"
    return f"{value:.2f} {unit}"


def format_range(low: float, high: float, unit: str) -> str:
    if unit == DEG:
        return f"{low:.0f}–{high:.0f}{DEG}"
    return f"{low:.2f}–{high:.2f} {unit}"


def format_deviation(dev: float, unit: str) -> str:
    if not math.isfinite(dev):
        return "n/a"
    if unit == DEG:
        return f"{dev:+.0f}{DEG}"
    return f"{dev:+.2f} {unit}"


def _adverb(severity: float) -> str:
    if severity < 0.25:
        return " slightly"
    if severity < 0.7:
        return ""
    return " significantly"


def build_correction(j: JointResult) -> Correction:
    direction = "low" if j.status == STATUS_LOW else "high"
    action = j.too_low if direction == "low" else j.too_high
    sev = j.severity
    if action:
        sentence = f"{action}{_adverb(sev)}."
    else:  # no instruction defined for that direction: describe it neutrally
        what = "angle" if j.unit == DEG else "distance"
        sentence = (f"Increase this {what}." if direction == "low" else f"Reduce this {what}.")
    text = (f"{j.label}: {format_value(j.value, j.unit)}. "
            f"Target: {format_range(j.low, j.high, j.unit)}. {sentence}")
    return Correction(j.key, j.label, j.value, j.low, j.high, j.deviation, j.unit, direction, sev,
                      action, sentence, text, j.joints)


def generate_corrections(joint_results: Sequence[JointResult], max_items: int = 3) -> List[Correction]:
    """Most important corrections first (weighted by how far outside the target each joint is)."""
    candidates = [
        j for j in joint_results
        if j.status in (STATUS_LOW, STATUS_HIGH) and math.isfinite(j.score) and j.score < CORRECTION_SCORE_MAX
    ]
    candidates.sort(key=lambda j: j.weight * j.severity, reverse=True)
    return [build_correction(j) for j in candidates[:max_items]]


def positive_feedback(joint_results: Sequence[JointResult], max_items: int = 2) -> List[str]:
    ok = [j for j in joint_results if j.status == STATUS_OK]
    ok.sort(key=lambda j: j.weight, reverse=True)
    return [f"{j.label} is in range ({format_value(j.value, j.unit)})." for j in ok[:max_items]]
