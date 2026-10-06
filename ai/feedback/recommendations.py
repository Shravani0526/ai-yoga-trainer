"""Rule-based recommendation engine (transparent: every recommendation names its trigger)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from ai.personalization.difficulty import LEVEL_ORDER, recommend_level
from ai.personalization.progress import prepare_sessions, trend_slope
from ai.personalization.recurring import describe_direction
from ai.yoga.poses import get_pose, list_poses


@dataclass
class Recommendation:
    category: str      # Focus / Stability / Symmetry / Progression / Next pose / Consistency / Getting started
    title: str
    detail: str
    priority: int      # 1 = most important
    reason: str        # the data that triggered the rule

    def as_dict(self) -> dict:
        return {"priority": self.priority, "category": self.category, "title": self.title,
                "detail": self.detail, "why": self.reason}


def session_feedback(summary: Dict) -> List[str]:
    """Short plain-language feedback for a just-finished session."""
    msgs: List[str] = []
    if summary.get("detected_frames", 0) < 10:
        return ["Very few frames were analysed - check that your whole body is visible and well lit."]
    score = summary.get("avg_score", 0.0)
    msgs.append(f"Average pose score {score:.0f}/100; best moment {summary.get('max_score', 0):.0f}/100.")
    held, target = summary.get("held_s", 0.0), summary.get("hold_target_s", 0.0)
    if summary.get("completed"):
        msgs.append(f"Hold target reached: {held:.0f}s of correct, stable holding (target {target:.0f}s).")
    else:
        msgs.append(f"You held the pose correctly for {held:.0f}s of the {target:.0f}s target.")
    errors = summary.get("errors", [])
    worst = [e for e in errors if e["bad_frac"] >= 0.25][:2]
    for e in worst:
        way = "too small / low" if e["direction"] == "low" else "too large / high"
        msgs.append(f"{e['label']} was out of range for {100 * e['bad_frac']:.0f}% of the session (usually {way}).")
    if summary.get("avg_stability", 100) < 60:
        msgs.append("Stability was low - slow your breathing and keep your gaze on one fixed point.")
    return msgs


def build_recommendations(sessions_df: pd.DataFrame, recurring_df: pd.DataFrame,
                          levels: Optional[Dict[str, str]] = None,
                          today: Optional[datetime] = None) -> List[Recommendation]:
    """Create prioritised recommendations from the user's whole history."""
    levels = levels or {}
    recs: List[Recommendation] = []
    today = today or datetime.now(timezone.utc)

    if sessions_df is None or sessions_df.empty:
        return [
            Recommendation("Getting started", "Calibrate first",
                           "Hold your most comfortable version of a pose for a few seconds on the Calibration page so the "
                           "target ranges fit your body.", 1, "No sessions recorded yet."),
            Recommendation("Getting started", "Start with T Pose or Tree Pose",
                           "They have the simplest geometry and give the fastest feedback. Face the camera and keep your "
                           "whole body visible.", 2, "No sessions recorded yet."),
        ]

    p = prepare_sessions(sessions_df)

    # 1. Recurring errors -> focused drills
    if recurring_df is not None and not recurring_df.empty:
        for i, row in recurring_df.head(3).iterrows():
            way = describe_direction(row["direction"], row["unit"])
            recs.append(Recommendation(
                "Focus", f"{row['pose']}: work on your {row['label'].lower()}",
                (row["tip"] or "Slow down and check this joint in a mirror.") +
                f" It is typically {way}.",
                1 + i,
                f"Out of range in {row['sessions_affected']} of the last {row['sessions_considered']} sessions "
                f"(about {row['avg_bad_pct']:.0f}% of frames)."))

    recent = p.tail(5)
    # 2. Stability
    if recent["avg_stability"].mean() < 60:
        recs.append(Recommendation(
            "Stability", "Improve steadiness",
            "Pick a fixed point (drishti) at eye level, breathe slowly through the nose and engage the legs. "
            "Practise with a wall or chair nearby for safety.", 4,
            f"Average stability over your last {len(recent)} sessions is {recent['avg_stability'].mean():.0f}/100."))

    # 3. Symmetry (only meaningful for symmetric poses)
    sym_poses = [n for n in list_poses() if get_pose(n).symmetric]
    sym = p[p["pose"].isin(sym_poses)].tail(5)
    if len(sym) and sym["avg_symmetry"].mean() < 70:
        recs.append(Recommendation(
            "Symmetry", "Even out left and right",
            "Practise in front of a mirror or film yourself from the front; compare the left and right elbow, "
            "shoulder and knee positions.", 5,
            f"Average symmetry in T Pose / Chair Pose is {sym['avg_symmetry'].mean():.0f}/100."))

    # 4. Adaptive difficulty per pose
    for pose, g in p.groupby("pose"):
        current = levels.get(pose, "Beginner")
        new_level, reason = recommend_level(g, current)
        if new_level != current:
            direction = "up" if LEVEL_ORDER.index(new_level) > LEVEL_ORDER.index(current) else "down"
            recs.append(Recommendation(
                "Progression", f"{pose}: move {direction} to {new_level}",
                reason, 3, f"Current level {current}; last two sessions: "
                           + ", ".join(f"{s:.0f}" for s in g.sort_values('started_at').tail(2)['avg_score'])))

    # 5. Trend warning
    for pose, g in p.groupby("pose"):
        if len(g) >= 4 and trend_slope(g["avg_score"]) < -2.0:
            recs.append(Recommendation(
                "Progression", f"{pose}: scores are dropping",
                "Rest, re-calibrate and repeat the pose at a lower difficulty before pushing again.", 4,
                f"Score trend {trend_slope(g['avg_score']):+.1f} points per session over {len(g)} sessions."))

    # 6. Next pose to practise
    practised = set(p["pose"])
    never = [n for n in list_poses() if n not in practised]
    if never:
        recs.append(Recommendation("Next pose", f"Try {never[0]}",
                                   get_pose(never[0]).benefits or get_pose(never[0]).description, 6,
                                   f"{never[0]} has not been practised yet."))
    else:
        weakest = p.groupby("pose")["avg_score"].mean().sort_values().index[0]
        recs.append(Recommendation("Next pose", f"Revisit {weakest}",
                                   f"{weakest} is currently your lowest-scoring pose - one more session with the "
                                   "corrections visible will help most.", 6,
                                   f"Average score {p[p['pose'] == weakest]['avg_score'].mean():.0f}/100."))

    # 7. Consistency
    last_day = p["started_at"].max()
    days_since = (today - last_day.to_pydatetime()).days
    if days_since >= 3:
        recs.append(Recommendation("Consistency", "Short daily sessions",
                                   "Five focused minutes a day beat one long session per week.", 7,
                                   f"Your last session was {days_since} days ago."))

    recs.sort(key=lambda r: r.priority)
    return recs
