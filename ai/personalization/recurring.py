"""Recurring-error detection across sessions.

A joint rule is "recurring" when, over the user's most recent sessions of a pose, it was
out of range for a meaningful share of frames in at least two sessions, and in at least half
of the sessions in which it was measured.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ai.yoga.poses import find_criterion

MIN_FRAMES = 15           # a session needs this many measured frames for the joint to count
BAD_FRAC_THRESHOLD = 0.30  # share of frames out of range for a session to be "affected"
MIN_AFFECTED_SESSIONS = 2
MIN_AFFECTED_SHARE = 0.5

RESULT_COLUMNS = ["pose", "key", "label", "sessions_considered", "sessions_affected",
                  "avg_bad_pct", "mean_deviation", "direction", "unit", "severity", "tip"]


def detect_recurring_errors(errors_df: pd.DataFrame, sessions_df: Optional[pd.DataFrame] = None,
                            window: int = 6) -> pd.DataFrame:
    """Return recurring joint errors.

    ``errors_df`` needs columns: session_id, pose, key, label, unit, frames_total, frames_bad,
    bad_frac, mean_deviation, direction, started_at. Only the last ``window`` sessions of each
    pose are considered.
    """
    if errors_df is None or errors_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = errors_df.copy()
    df = df[df["frames_total"] >= MIN_FRAMES]
    if df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    rows = []
    for pose, pose_df in df.groupby("pose"):
        recent_ids = (pose_df[["session_id", "started_at"]].drop_duplicates()
                      .sort_values("started_at").tail(window)["session_id"].tolist())
        pose_df = pose_df[pose_df["session_id"].isin(recent_ids)]
        for key, g in pose_df.groupby("key"):
            considered = g["session_id"].nunique()
            affected = g[g["bad_frac"] >= BAD_FRAC_THRESHOLD]
            n_aff = affected["session_id"].nunique()
            if n_aff < MIN_AFFECTED_SESSIONS or n_aff / considered < MIN_AFFECTED_SHARE:
                continue
            weights = affected["frames_bad"].clip(lower=1)
            mean_dev = float((affected["mean_deviation"] * weights).sum() / weights.sum())
            low_w = float(weights[affected["direction"] == "low"].sum())
            high_w = float(weights[affected["direction"] == "high"].sum())
            direction = "low" if low_w > high_w else "high"
            avg_bad = float(affected["bad_frac"].mean())
            crit = find_criterion(pose, key)
            tip = crit.tip if crit else ""
            rows.append({
                "pose": pose, "key": key, "label": g["label"].iloc[-1],
                "sessions_considered": int(considered), "sessions_affected": int(n_aff),
                "avg_bad_pct": round(100 * avg_bad, 1), "mean_deviation": round(mean_dev, 2),
                "direction": direction, "unit": g["unit"].iloc[-1],
                "severity": round(n_aff / considered * avg_bad * 100, 1), "tip": tip,
            })
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    out = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return out.sort_values("severity", ascending=False).reset_index(drop=True)


def describe_direction(direction: str, unit: str) -> str:
    """Human wording for the direction of a recurring error."""
    if direction == "low":
        return "usually too small / too low" if unit != "°" else "angle usually too small (too bent / too low)"
    return "usually too large / too high" if unit != "°" else "angle usually too large (too open / too straight)"
