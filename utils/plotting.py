"""Matplotlib figures used by the Streamlit app and the Colab notebook."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# NOTE: no matplotlib.use(...) here on purpose - Colab / Jupyter need their inline backend, and
# headless servers (Streamlit Cloud) automatically fall back to the non-interactive Agg backend.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai.pose.keypoints import SKELETON_EDGES

TEAL, AMBER, CORAL, NAVY, GREY = "#2a9d8f", "#e9a23b", "#e76f51", "#264653", "#9aa5b1"
POSE_COLORS = {"T Pose": "#2a9d8f", "Tree Pose": "#6a994e", "Warrior II": "#e76f51",
               "Chair Pose": "#457b9d", "Triangle Pose": "#9b5de5"}


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _empty(message: str, figsize=(7, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=GREY, fontsize=11)
    ax.axis("off")
    return fig


def timeline_figure(timeline: Sequence[Dict], hold_target_s: Optional[float] = None):
    """Score / alignment / stability over time; shaded where the pose was correct."""
    if not timeline:
        return _empty("No frames recorded")
    df = pd.DataFrame(timeline)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for col, color, lw, alpha, label in (("score", TEAL, 2, 1.0, "Overall score"),
                                         ("alignment", NAVY, 1.2, 0.8, "Alignment"),
                                         ("stability", AMBER, 1.2, 0.9, "Stability")):
        if col in df:
            ax.plot(df["t"], df[col], color=color, lw=lw, alpha=alpha, label=label)
    if "held" not in df:
        df["held"] = 0.0
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="lower center")
    _style(ax, "Score over time", ylabel="Score (0-100)")
    ax2.plot(df["t"], df["held"], color=TEAL, lw=2)
    ax2.fill_between(df["t"], 0, df["held"], color=TEAL, alpha=0.15)
    if hold_target_s:
        ax2.axhline(hold_target_s, color=CORAL, ls="--", lw=1)
        ax2.text(df["t"].iloc[0], hold_target_s, " hold target", color=CORAL, fontsize=8, va="bottom")
    _style(ax2, "Hold timer (counts only when correct and stable)", "Time (s)", "Held (s)")
    fig.tight_layout()
    return fig


def joint_error_figure(errors: Sequence[Dict], top: int = 10):
    """Horizontal bars: share of frames each joint was outside its target range."""
    if not errors:
        return _empty("No joint statistics yet")
    df = pd.DataFrame(errors).sort_values("bad_frac", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.42 * len(df) + 1)))
    colors = [CORAL if v >= 0.3 else (AMBER if v >= 0.1 else TEAL) for v in df["bad_frac"]]
    ax.barh(df["label"], df["bad_frac"] * 100, color=colors)
    ax.set_xlim(0, 100)
    _style(ax, "Time outside target range (per joint)", "% of frames")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def progress_figure(df: pd.DataFrame):
    """Average score per session with a 3-session moving average, coloured by pose."""
    if df is None or df.empty:
        return _empty("No sessions yet - complete a training session first")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for pose, g in df.groupby("pose"):
        ax.scatter(g["session_no"], g["avg_score"], s=36, color=POSE_COLORS.get(pose, GREY), label=pose, zorder=3)
    ax.plot(df["session_no"], df["score_ma3"], color=NAVY, lw=2, label="3-session average")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="lower right")
    _style(ax, "Score per session", "Session #", "Average score")
    fig.tight_layout()
    return fig


def hold_time_figure(df: pd.DataFrame):
    if df is None or df.empty:
        return _empty("No sessions yet")
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = [POSE_COLORS.get(p, GREY) for p in df["pose"]]
    ax.bar(df["session_no"], df["held_s"], color=colors)
    ax.plot(df["session_no"], df["hold_target_s"], color=CORAL, ls="--", lw=1.2, label="hold target")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Correct hold time per session", "Session #", "Seconds held")
    fig.tight_layout()
    return fig


def recurring_figure(rec_df: pd.DataFrame):
    if rec_df is None or rec_df.empty:
        return _empty("No recurring errors detected")
    d = rec_df.copy()
    d["name"] = d["pose"] + " - " + d["label"]
    d = d.sort_values("severity", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.45 * len(d) + 1)))
    ax.barh(d["name"], d["avg_bad_pct"], color=CORAL)
    for y, (aff, con) in enumerate(zip(d["sessions_affected"], d["sessions_considered"])):
        ax.text(1, y, f" {aff}/{con} sessions", va="center", fontsize=8, color="white")
    ax.set_xlim(0, 100)
    _style(ax, "Recurring errors (avg % of frames out of range)", "% of frames")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def metric_timeseries_figure(times: Sequence[float], series: Dict[str, Sequence[float]],
                             ranges: Optional[Dict[str, tuple]] = None, title: str = "Joint angles over time"):
    """Plot several measured metrics; optional target ranges are drawn as bands."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    palette = [TEAL, CORAL, NAVY, AMBER, "#6a994e", "#9b5de5"]
    for i, (name, vals) in enumerate(series.items()):
        col = palette[i % len(palette)]
        ax.plot(times, vals, lw=1.6, color=col, label=name)
        if ranges and name in ranges:
            lo, hi = ranges[name]
            ax.axhspan(lo, hi, color=col, alpha=0.10)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    _style(ax, title, "Time (s)", "Degrees")
    fig.tight_layout()
    return fig


def skeleton_figure(coords: np.ndarray, title: str = "", ax=None):
    """Draw a normalised skeleton (x right, y down) - used for synthetic self-checks."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3, 4))
    else:
        fig = ax.figure
    for a, b in SKELETON_EDGES:
        if not (np.isnan(coords[a]).any() or np.isnan(coords[b]).any()):
            ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]], color=NAVY, lw=2)
    ok = ~np.isnan(coords).any(axis=1)
    ax.scatter(coords[ok, 0], coords[ok, 1], s=22, color=TEAL, zorder=3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, color=NAVY)
    ax.axis("off")
    if own:
        fig.tight_layout()
    return fig
