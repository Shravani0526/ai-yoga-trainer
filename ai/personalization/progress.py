"""Progress tracking with pandas: trends, streaks and per-pose summaries."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd


def prepare_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and add convenience columns (session number, moving average)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["started_at"] = pd.to_datetime(out["started_at"], utc=True, errors="coerce")
    out = out.sort_values("started_at").reset_index(drop=True)
    out["session_no"] = np.arange(1, len(out) + 1)
    out["date"] = out["started_at"].dt.date
    out["score_ma3"] = out["avg_score"].rolling(3, min_periods=1).mean()
    return out


def trend_slope(values: pd.Series) -> float:
    """Least-squares slope in score points per session (0 when fewer than 3 sessions)."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(v) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])


def practice_streak(dates, today: Optional[date] = None) -> int:
    """Consecutive practice days ending today (or yesterday)."""
    days = sorted({d for d in dates if d is not None})
    if not days:
        return 0
    today = today or datetime.now(timezone.utc).date()
    if days[-1] < today - timedelta(days=1):
        return 0
    streak, cursor = 0, days[-1]
    day_set = set(days)
    while cursor in day_set:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def overall_summary(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"sessions": 0, "total_hold_min": 0.0, "best_score": 0.0, "avg_score": 0.0,
                "improvement": 0.0, "streak_days": 0, "completion_rate": 0.0}
    p = prepare_sessions(df)
    n = len(p)
    last5, prev5 = p.tail(5), p.iloc[max(0, n - 10):max(0, n - 5)]
    improvement = float(last5["avg_score"].mean() - prev5["avg_score"].mean()) if len(prev5) else 0.0
    return {
        "sessions": int(n),
        "total_hold_min": float(p["held_s"].sum() / 60.0),
        "best_score": float(p["avg_score"].max()),
        "avg_score": float(p["avg_score"].mean()),
        "improvement": improvement,
        "streak_days": practice_streak(p["date"].tolist()),
        "completion_rate": float(p["completed"].astype(int).mean() * 100.0),
    }


def per_pose_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pose: sessions, best / average / latest score, trend and total hold time."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Pose", "Sessions", "Best score", "Average score",
                                     "Latest score", "Trend / session", "Hold time (s)", "Completed"])
    p = prepare_sessions(df)
    rows = []
    for pose, g in p.groupby("pose"):
        rows.append({
            "Pose": pose,
            "Sessions": len(g),
            "Best score": round(g["avg_score"].max(), 1),
            "Average score": round(g["avg_score"].mean(), 1),
            "Latest score": round(g["avg_score"].iloc[-1], 1),
            "Trend / session": round(trend_slope(g["avg_score"]), 2),
            "Hold time (s)": round(g["held_s"].sum(), 1),
            "Completed": int(g["completed"].astype(int).sum()),
        })
    return pd.DataFrame(rows).sort_values("Sessions", ascending=False).reset_index(drop=True)


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average score and hold time per ISO week."""
    p = prepare_sessions(df)
    if p.empty:
        return pd.DataFrame(columns=["week", "sessions", "avg_score", "held_s"])
    p["week"] = p["started_at"].dt.tz_convert("UTC").dt.strftime("%G-W%V")
    return (p.groupby("week").agg(sessions=("session_no", "count"), avg_score=("avg_score", "mean"),
                                  held_s=("held_s", "sum")).reset_index())
