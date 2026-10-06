"""Difficulty levels and adaptive difficulty.

A level changes (a) how wide the target ranges are, (b) how good a pose must be to count
as "correct" for the hold timer and (c) the hold target. Adaptive difficulty promotes or
demotes a user *per pose* using simple, explainable rules on their recent sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ai.yoga.criteria import Criterion

LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]


@dataclass(frozen=True)
class LevelConfig:
    name: str
    range_expand: float      # fraction of each rule's tolerance added to (or removed from) the range
    min_alignment: float     # alignment score needed to count as "correct"
    min_stability: float     # stability score needed to count as "stable"
    hold_target_s: float
    strict_hold: bool        # target must be reached in ONE uninterrupted hold
    description: str


LEVELS = {
    "Beginner": LevelConfig("Beginner", +0.30, 60.0, 45.0, 10.0, False,
                            "Wider target ranges, shorter hold, interruptions allowed."),
    "Intermediate": LevelConfig("Intermediate", 0.0, 72.0, 60.0, 20.0, False,
                                "Standard target ranges and a 20-second hold."),
    "Advanced": LevelConfig("Advanced", -0.15, 82.0, 72.0, 30.0, True,
                            "Tighter ranges and one uninterrupted 30-second hold."),
}


def get_level(name: str) -> LevelConfig:
    return LEVELS.get(name, LEVELS["Beginner"])


def apply_difficulty(criteria: Sequence[Criterion], level_name: str) -> List[Criterion]:
    """Return criteria whose ranges are widened (Beginner) or tightened (Advanced)."""
    cfg = get_level(level_name)
    out: List[Criterion] = []
    for c in criteria:
        delta = cfg.range_expand * c.tol
        low, high = c.low - delta, c.high + delta
        if low > high:                       # narrowing never inverts the range
            mid = (c.low + c.high) / 2.0
            low = high = mid
        if c.is_angle:                       # angles stay physically meaningful
            low, high = max(0.0, low), min(180.0, high)
        else:
            low = max(0.0, low)
        out.append(c.with_range(low, high))
    return out


def recommend_level(history: pd.DataFrame, current_level: str,
                    min_sessions: int = 2) -> Tuple[str, str]:
    """Return (new_level, reason) from a pose's recent sessions (newest last).

    Rules (documented, not learned):
        promote  : the last two sessions both scored >= 80 and completed the hold
        demote   : the last two sessions both scored < 45
        otherwise: stay
    """
    idx = LEVEL_ORDER.index(current_level) if current_level in LEVEL_ORDER else 0
    if history is None or len(history) < min_sessions:
        return LEVEL_ORDER[idx], "Not enough sessions yet - complete at least two sessions of this pose."
    last = history.sort_values("started_at").tail(min_sessions)
    scores = last["avg_score"].astype(float)
    completed = last["completed"].astype(int)
    if (scores >= 80).all() and (completed == 1).all():
        if idx < len(LEVEL_ORDER) - 1:
            return LEVEL_ORDER[idx + 1], "Your last two sessions scored 80+ and finished the hold - level up!"
        return LEVEL_ORDER[idx], "You are already at the highest level and performing very well."
    if (scores < 45).all():
        if idx > 0:
            return LEVEL_ORDER[idx - 1], "Your last two sessions scored below 45 - an easier level will help you build up."
        return LEVEL_ORDER[idx], "Keep practising at Beginner level; use the corrections to improve."
    return LEVEL_ORDER[idx], "Keep practising at this level until you score 80+ and complete the hold twice in a row."
