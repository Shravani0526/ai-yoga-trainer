"""Reference illustrations: how each pose should look, with the target joint angles.

The figures are drawn by this project's own code from the same synthetic skeletons used in the
tests (so they always match the rules in ai/yoga/poses.py). They are simplified stick figures,
not photographs - the aim is to show the *shape* of the pose and its key angles.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai.pose.keypoints import KP
from ai.yoga.poses import criteria_for, get_pose
from utils.synthetic import make_pose

BG = (250, 249, 244)            # RGB
LEFT_COL = (42, 157, 143)       # teal  = subject's LEFT side
RIGHT_COL = (38, 70, 83)        # navy  = subject's RIGHT side
TORSO_COL = (205, 225, 222)
TEXT_COL = (38, 70, 83)

_JOINT_METRICS = ("knee", "elbow", "shoulder", "hip")


def _limb_color(idx: int) -> Tuple[int, int, int]:
    if idx == 0:
        return TEXT_COL
    return LEFT_COL if idx % 2 == 1 else RIGHT_COL


def _labels_for(pose: str, variant: str, max_labels: int = 4) -> List[Tuple[int, str]]:
    """(keypoint index, 'low-high deg') for the most important joint-angle rules."""
    crits = [c for c in criteria_for(pose, variant)
             if c.is_angle and c.metric.split("_")[-1] in _JOINT_METRICS and c.metric.split("_")[0] in ("left", "right")]
    crits.sort(key=lambda c: c.weight, reverse=True)
    out: List[Tuple[int, str]] = []
    seen = set()
    for c in crits:
        idx = KP[c.metric]
        if idx in seen:
            continue
        seen.add(idx)
        out.append((idx, f"{c.low:.0f}-{c.high:.0f} deg"))
        if len(out) == max_labels:
            break
    return out


def render_reference(pose: str, variant: str, width: int = 360, height: int = 460,
                     mirror: bool = False, show_targets: bool = True) -> np.ndarray:
    """Return an RGB image (height, width, 3) of the pose.

    ``mirror=True`` flips the picture so it matches a mirrored webcam view (your right side on
    the right of the screen).
    """
    spec = get_pose(pose)
    v = variant if variant in spec.variants else spec.variant_names[0]
    coords = make_pose(pose, "left" if v == "auto" else v)

    img = np.full((height, width, 3), BG, dtype=np.uint8)
    head_r = 0.22
    xs = np.concatenate([coords[:, 0], [coords[KP["nose"], 0] - head_r, coords[KP["nose"], 0] + head_r]])
    ys = np.concatenate([coords[:, 1], [coords[KP["nose"], 1] - head_r, coords[KP["nose"], 1] + head_r]])
    margin_x, margin_top, margin_bot = 0.20 * width, 0.08 * height, 0.14 * height
    scale = min((width - 2 * margin_x) / max(xs.max() - xs.min(), 1e-6),
                (height - margin_top - margin_bot) / max(ys.max() - ys.min(), 1e-6))
    ox = width / 2 - scale * (xs.max() + xs.min()) / 2
    oy = margin_top - scale * ys.min()
    pts = np.stack([coords[:, 0] * scale + ox, coords[:, 1] * scale + oy], axis=1)
    if mirror:
        pts[:, 0] = width - 1 - pts[:, 0]

    P = lambda i: (int(round(pts[i, 0])), int(round(pts[i, 1])))  # noqa: E731
    # floor line
    floor_y = int(max(pts[KP["left_ankle"], 1], pts[KP["right_ankle"], 1]) + 0.09 * scale)
    cv2.line(img, (int(0.1 * width), floor_y), (int(0.9 * width), floor_y), (200, 205, 205), 2, cv2.LINE_AA)

    # torso
    torso = np.array([P(KP["left_shoulder"]), P(KP["right_shoulder"]), P(KP["right_hip"]), P(KP["left_hip"])], np.int32)
    cv2.fillPoly(img, [torso], TORSO_COL, cv2.LINE_AA)
    cv2.polylines(img, [torso], True, (150, 185, 180), 2, cv2.LINE_AA)

    # head
    nose = pts[KP["nose"]]
    cv2.circle(img, (int(nose[0]), int(nose[1] - 0.05 * scale)), int(head_r * scale * 0.9), (236, 226, 214), -1, cv2.LINE_AA)
    cv2.circle(img, (int(nose[0]), int(nose[1] - 0.05 * scale)), int(head_r * scale * 0.9), TEXT_COL, 2, cv2.LINE_AA)

    # limbs (thick, coloured by side)
    thick = max(6, int(0.045 * scale))
    for side in ("left", "right"):
        for a, b in ((f"{side}_shoulder", f"{side}_elbow"), (f"{side}_elbow", f"{side}_wrist"),
                     (f"{side}_hip", f"{side}_knee"), (f"{side}_knee", f"{side}_ankle")):
            cv2.line(img, P(KP[a]), P(KP[b]), _limb_color(KP[a]), thick, cv2.LINE_AA)
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "right_elbow",
                 "left_knee", "right_knee", "left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        i = KP[name]
        cv2.circle(img, P(i), thick // 2 + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, P(i), thick // 2 + 3, _limb_color(i), 2, cv2.LINE_AA)

    # target angle labels (a label that would overlap an earlier one is skipped)
    if show_targets:
        center = pts[[KP["left_hip"], KP["right_hip"]]].mean(axis=0)
        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.46
        placed: List[Tuple[int, int, int, int]] = []
        for idx, text in _labels_for(pose, v, max_labels=6):
            p = pts[idx]
            direction = p - center
            n = np.linalg.norm(direction)
            direction = direction / n if n > 1e-6 else np.array([1.0, 0.0])
            (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
            anchor = p + direction * 28 + np.array([0, 4])
            x = int(np.clip(anchor[0] - (tw / 2 if abs(direction[0]) < 0.4 else (0 if direction[0] > 0 else tw)), 6, width - tw - 8))
            y = int(np.clip(anchor[1], th + 6, height - 8))
            box = (x - 4, y - th - 4, x + tw + 4, y + 5)
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
                continue
            placed.append(box)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), _limb_color(idx), 1)
            cv2.putText(img, text, (x, y), font, fs, TEXT_COL, 1, cv2.LINE_AA)
            if len(placed) == 4:
                break

    # L / R markers at the hands so users know which side is which
    for side, col in (("left", LEFT_COL), ("right", RIGHT_COL)):
        w = pts[KP[f"{side}_ankle"]]
        cv2.putText(img, "L" if side == "left" else "R", (int(w[0]) - 6, int(w[1]) + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
    return img


def reference_caption(mirror: bool = True) -> str:
    view = "like a mirror (your right side is on the right of the screen)" if mirror else "as the camera sees you"
    return (f"Teal = your LEFT side, dark blue = your RIGHT side; shown {view}. "
            "Boxes show the target angle ranges (deg).")
