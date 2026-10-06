"""Angle / distance calculations on (17, 2) keypoint arrays.

All functions propagate NaN: if a required keypoint is missing the metric is NaN
and the alignment layer simply skips it (it is never guessed).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ai.pose.keypoints import KP


def _valid(*pts) -> bool:
    return all(p is not None and not np.any(np.isnan(p)) for p in pts)


def angle_3pts(a, b, c) -> float:
    """Interior angle in degrees at vertex ``b`` formed by points a-b-c (0..180)."""
    if not _valid(a, b, c):
        return float("nan")
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def distance(a, b) -> float:
    if not _valid(a, b):
        return float("nan")
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def midpoint(a, b):
    if not _valid(a, b):
        return np.array([np.nan, np.nan])
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def tilt_from_horizontal(a, b) -> float:
    """Absolute tilt (0..90 deg) of the line a-b relative to the horizontal axis."""
    if not _valid(a, b):
        return float("nan")
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return float("nan")
    ang = abs(math.degrees(math.atan2(dy, dx)))
    return 180.0 - ang if ang > 90.0 else ang


def tilt_from_vertical(a, b) -> float:
    """Absolute tilt (0..90 deg) of the line a-b relative to the vertical axis."""
    t = tilt_from_horizontal(a, b)
    return float("nan") if math.isnan(t) else 90.0 - t


def compute_proportions(coords: np.ndarray) -> Dict[str, float]:
    """Body proportions relative to torso length (torso = mid-shoulder to mid-hip)."""
    p = coords
    torso = distance(midpoint(p[KP["left_shoulder"]], p[KP["right_shoulder"]]),
                     midpoint(p[KP["left_hip"]], p[KP["right_hip"]]))
    out = {"leg_ratio": float("nan"), "arm_ratio": float("nan")}
    if not math.isfinite(torso) or torso < 1e-9:
        return out
    legs, arms = [], []
    for side in ("left", "right"):
        thigh = distance(p[KP[f"{side}_hip"]], p[KP[f"{side}_knee"]])
        shin = distance(p[KP[f"{side}_knee"]], p[KP[f"{side}_ankle"]])
        upper = distance(p[KP[f"{side}_shoulder"]], p[KP[f"{side}_elbow"]])
        fore = distance(p[KP[f"{side}_elbow"]], p[KP[f"{side}_wrist"]])
        if math.isfinite(thigh) and math.isfinite(shin):
            legs.append((thigh + shin) / torso)
        if math.isfinite(upper) and math.isfinite(fore):
            arms.append((upper + fore) / torso)
    if legs:
        out["leg_ratio"] = float(np.mean(legs))
    if arms:
        out["arm_ratio"] = float(np.mean(arms))
    return out


def compute_metrics(coords: np.ndarray) -> Dict[str, float]:
    """Compute every joint angle / distance used by the yoga rules.

    ``coords`` is a (17, 2) array in image orientation (x right, y DOWN). Distances are
    expressed in torso lengths, so the metrics do not depend on how far the person is
    from the camera.
    """
    p = coords
    g = lambda name: p[KP[name]]  # noqa: E731
    m: Dict[str, float] = {}

    for side in ("left", "right"):
        sh, el, wr = g(f"{side}_shoulder"), g(f"{side}_elbow"), g(f"{side}_wrist")
        hip, kn, an = g(f"{side}_hip"), g(f"{side}_knee"), g(f"{side}_ankle")
        m[f"{side}_elbow"] = angle_3pts(sh, el, wr)
        m[f"{side}_shoulder"] = angle_3pts(hip, sh, el)   # arm raise angle
        m[f"{side}_hip"] = angle_3pts(sh, hip, kn)
        m[f"{side}_knee"] = angle_3pts(hip, kn, an)

    mid_sh = midpoint(g("left_shoulder"), g("right_shoulder"))
    mid_hip = midpoint(g("left_hip"), g("right_hip"))
    torso = distance(mid_sh, mid_hip)
    m["torso_len"] = torso

    # Torso lean: angle between hip->shoulder vector and "up" (0 deg = upright).
    if math.isfinite(torso) and torso > 1e-9:
        v = mid_sh - mid_hip
        cos = float(np.clip(np.dot(v, np.array([0.0, -1.0])) / torso, -1.0, 1.0))
        m["torso_lean"] = math.degrees(math.acos(cos))
    else:
        m["torso_lean"] = float("nan")

    m["shoulder_tilt"] = tilt_from_horizontal(g("left_shoulder"), g("right_shoulder"))
    m["hip_tilt"] = tilt_from_horizontal(g("left_hip"), g("right_hip"))

    scale = torso if math.isfinite(torso) and torso > 1e-9 else float("nan")
    norm = lambda d: d / scale if math.isfinite(d) else float("nan")  # noqa: E731

    m["ankle_gap"] = norm(distance(g("left_ankle"), g("right_ankle")))
    m["wrist_gap"] = norm(distance(g("left_wrist"), g("right_wrist")))
    # Positive when that foot is raised above the other foot (image y grows downwards).
    la, ra = g("left_ankle"), g("right_ankle")
    if _valid(la, ra):
        m["left_foot_lift"] = norm(float(ra[1] - la[1]))
        m["right_foot_lift"] = norm(float(la[1] - ra[1]))
    else:
        m["left_foot_lift"] = m["right_foot_lift"] = float("nan")
    m["left_wrist_ankle"] = norm(distance(g("left_wrist"), g("left_ankle")))
    m["right_wrist_ankle"] = norm(distance(g("right_wrist"), g("right_ankle")))

    # Hands height: 0 = at hip line, 1 = at shoulder line.
    wrist_mid = midpoint(g("left_wrist"), g("right_wrist"))
    if _valid(wrist_mid, mid_hip) and math.isfinite(scale):
        m["wrist_level"] = float((mid_hip[1] - wrist_mid[1]) / scale)
    else:
        m["wrist_level"] = float("nan")

    # Is the line joining the two wrists vertical? (Triangle: arms stacked in one line.)
    lw, rw = g("left_wrist"), g("right_wrist")
    m["wrist_line_vertical"] = tilt_from_vertical(lw, rw)
    return m
