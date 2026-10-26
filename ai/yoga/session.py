"""Session recorder: aggregates per-frame results into a session summary.

The summary dictionary is exactly what the database stores and what the dashboards,
recurring-error detection and recommendation engine consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ai.yoga.alignment import STATUS_HIGH, STATUS_LOW, STATUS_MISSING, STATUS_OK, AlignmentResult

MAX_TIMELINE_POINTS = 300
BAD_SCORE_THRESHOLD = 80.0      # a joint counts as "off" in a frame when its own score is below this


def _finite(v: Optional[float]) -> bool:
    return v is not None and math.isfinite(v)


@dataclass
class _JointStats:
    label: str
    unit: str
    n: int = 0
    n_bad: int = 0
    n_low: int = 0
    n_high: int = 0
    dev_sum: float = 0.0
    score_sum: float = 0.0


class SessionRecorder:
    """Accumulates frames of one training session."""

    def __init__(self, pose: str, variant: str, difficulty: str, source: str = "webcam",
                 hold_target_s: float = 20.0):
        self.pose = pose
        self.variant = variant
        self.difficulty = difficulty
        self.source = source
        self.hold_target_s = hold_target_s
        self.reset()

    def reset(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.frames = 0
        self.detected = 0
        self.correct = 0
        self.first_t: Optional[float] = None
        self.last_t: Optional[float] = None
        self.sum_score = self.sum_align = self.sum_stab = self.sum_sym = 0.0
        self.n_score = self.n_align = self.n_stab = self.n_sym = 0
        self.max_score = 0.0
        self.timeline: List[Dict[str, Any]] = []
        self.joint_stats: Dict[str, _JointStats] = {}
        self.hold: Dict[str, Any] = {"held_s": 0.0, "best_streak_s": 0.0, "breaks": 0, "completed": False}
        self.variant_used: Optional[str] = None
        self.correction_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ recording
    def add_frame(self, t: float, detected: bool, alignment: Optional[AlignmentResult],
                  overall: float, stability: float, symmetry: float, is_correct: bool,
                  hold: Dict[str, Any], variant_used: Optional[str] = None,
                  correction_keys: Optional[List[str]] = None) -> None:
        self.frames += 1
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        self.hold = dict(hold)
        if variant_used:
            self.variant_used = variant_used
        if not detected or alignment is None:
            return
        self.detected += 1
        if is_correct:
            self.correct += 1
        if _finite(overall):
            self.sum_score += overall
            self.n_score += 1
            self.max_score = max(self.max_score, overall)
        if _finite(alignment.score):
            self.sum_align += alignment.score
            self.n_align += 1
        if _finite(stability):
            self.sum_stab += stability
            self.n_stab += 1
        if _finite(symmetry):
            self.sum_sym += symmetry
            self.n_sym += 1
        for j in alignment.joint_results:
            if j.status == STATUS_MISSING:
                continue
            st = self.joint_stats.setdefault(j.key, _JointStats(j.label, j.unit))
            st.n += 1
            st.score_sum += j.score
            if j.status in (STATUS_LOW, STATUS_HIGH) and j.score < BAD_SCORE_THRESHOLD:
                st.n_bad += 1
                st.dev_sum += j.deviation
                if j.status == STATUS_LOW:
                    st.n_low += 1
                else:
                    st.n_high += 1
        for key in correction_keys or []:
            self.correction_counts[key] = self.correction_counts.get(key, 0) + 1
        self.timeline.append({
            "t": round(t - (self.first_t or 0.0), 2),
            "score": round(overall, 1) if _finite(overall) else None,
            "alignment": round(alignment.score, 1) if _finite(alignment.score) else None,
            "stability": round(stability, 1) if _finite(stability) else None,
            "symmetry": round(symmetry, 1) if _finite(symmetry) else None,
            "correct": int(bool(is_correct)),
            "held": round(float(hold.get("held_s", 0.0)), 2),
        })

    # ------------------------------------------------------------------ summary
    def _downsample(self) -> List[Dict[str, Any]]:
        if len(self.timeline) <= MAX_TIMELINE_POINTS:
            return list(self.timeline)
        idx = np.linspace(0, len(self.timeline) - 1, MAX_TIMELINE_POINTS).astype(int)
        return [self.timeline[i] for i in idx]

    def error_rows(self, min_frames: int = 5) -> List[Dict[str, Any]]:
        rows = []
        for key, st in self.joint_stats.items():
            if st.n < min_frames:
                continue
            bad_frac = st.n_bad / st.n
            mean_dev = st.dev_sum / st.n_bad if st.n_bad else 0.0
            direction = "low" if st.n_low > st.n_high else ("high" if st.n_high > st.n_low else "none")
            rows.append({
                "key": key, "label": st.label, "unit": st.unit,
                "frames_total": st.n, "frames_bad": st.n_bad,
                "bad_frac": round(bad_frac, 4), "mean_deviation": round(mean_dev, 3),
                "direction": direction, "mean_score": round(st.score_sum / st.n, 2),
            })
        rows.sort(key=lambda r: r["bad_frac"], reverse=True)
        return rows

    def summary(self) -> Dict[str, Any]:
        avg = lambda s, n: round(s / n, 2) if n else 0.0  # noqa: E731
        duration = (self.last_t - self.first_t) if (self.first_t is not None and self.last_t is not None) else 0.0
        held = float(self.hold.get("held_s", 0.0))
        return {
            "pose": self.pose,
            "variant": self.variant_used or self.variant,
            "difficulty": self.difficulty,
            "source": self.source,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_s": round(duration, 2),
            "frames": self.frames,
            "detected_frames": self.detected,
            "correct_frames": self.correct,
            "held_s": round(held, 2),
            "best_streak_s": round(float(self.hold.get("best_streak_s", 0.0)), 2),
            "breaks": int(self.hold.get("breaks", 0)),
            "hold_target_s": float(self.hold_target_s),
            "completed": int(bool(self.hold.get("completed", False))),
            "avg_score": avg(self.sum_score, self.n_score),
            "avg_alignment": avg(self.sum_align, self.n_align),
            "avg_stability": avg(self.sum_stab, self.n_stab),
            "avg_symmetry": avg(self.sum_sym, self.n_sym),
            "max_score": round(self.max_score, 2),
            "timeline": self._downsample(),
            "errors": self.error_rows(),
        }
