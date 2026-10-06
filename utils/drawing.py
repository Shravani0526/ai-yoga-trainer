"""OpenCV drawing: skeleton overlay with per-joint status colours and a compact HUD.

OpenCV's built-in fonts only support ASCII, so text is converted with :func:`ascii_text`
(for example the degree sign becomes "deg").
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import numpy as np

from ai.pose.keypoints import SKELETON_EDGES
from ai.yoga.alignment import STATUS_HIGH, STATUS_LOW

GREEN = (80, 200, 120)     # BGR
AMBER = (0, 190, 255)
RED = (70, 70, 235)
WHITE = (245, 245, 245)
DARK = (40, 40, 40)
GREY = (170, 170, 170)


def ascii_text(text: str) -> str:
    return (text.replace("°", " deg").replace("–", "-").replace("—", "-")
            .replace("×", "x").encode("ascii", "ignore").decode("ascii"))


def _joint_colors(result) -> Dict[int, tuple]:
    """Map keypoint index -> colour from the alignment result (red = off, amber = slightly off)."""
    colors: Dict[int, tuple] = {}
    if result is None or result.alignment is None:
        return colors
    for j in result.alignment.joint_results:
        if j.status in (STATUS_LOW, STATUS_HIGH) and math.isfinite(j.score):
            col = RED if j.score < 55 else AMBER
            for idx in j.joints:
                if colors.get(idx) != RED:
                    colors[idx] = col
    return colors


def draw_skeleton(frame: np.ndarray, kps: np.ndarray, joint_colors: Optional[Dict[int, tuple]] = None,
                  min_conf: float = 0.25, base_color: tuple = GREEN) -> np.ndarray:
    """Draw bones and joints on ``frame`` in place. ``kps`` is (17, 3) in pixels."""
    joint_colors = joint_colors or {}
    h, w = frame.shape[:2]
    thick = max(2, int(round(min(h, w) / 220)))
    for a, b in SKELETON_EDGES:
        if kps[a, 2] >= min_conf and kps[b, 2] >= min_conf:
            col = RED if (joint_colors.get(a) == RED or joint_colors.get(b) == RED) else base_color
            cv2.line(frame, (int(kps[a, 0]), int(kps[a, 1])), (int(kps[b, 0]), int(kps[b, 1])), col, thick, cv2.LINE_AA)
    for i in range(len(kps)):
        if kps[i, 2] >= min_conf:
            col = joint_colors.get(i, base_color)
            cv2.circle(frame, (int(kps[i, 0]), int(kps[i, 1])), thick + 2, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(kps[i, 0]), int(kps[i, 1])), thick + 2, WHITE, 1, cv2.LINE_AA)
    return frame


def _panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), DARK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put(frame, text, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(frame, ascii_text(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def annotate_frame(frame_bgr: np.ndarray, result, mirror: bool = False, show_hud: bool = True,
                   min_conf: float = 0.25) -> np.ndarray:
    """Return an annotated copy of ``frame_bgr`` (skeleton + HUD). ``result`` is a FrameResult."""
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    kps = None if result is None else result.keypoints_display
    if mirror:
        frame = cv2.flip(frame, 1)
        if kps is not None:
            kps = kps.copy()
            kps[:, 0] = (w - 1) - kps[:, 0]
    scale = max(0.45, min(w, 1280) / 1100.0)

    if kps is not None and result is not None and result.detected:
        draw_skeleton(frame, kps, _joint_colors(result), min_conf)
    elif kps is not None:
        draw_skeleton(frame, kps, None, min_conf, GREY)

    if not show_hud or result is None:
        return frame

    pad = int(10 * scale)
    line = int(26 * scale)
    _panel(frame, 0, 0, min(w, int(330 * scale) + pad), pad * 2 + line * 5)
    pose_text = result.detected_pose if result.detected else "No body detected"
    _put(frame, f"Pose: {pose_text}", (pad, pad + line * 1 - 6), 0.62 * scale / 0.55 * 0.9, WHITE, 1)
    if result.detected and result.breakdown is not None:
        sc = result.overall
        col = GREEN if sc >= 75 else (AMBER if sc >= 50 else RED)
        _put(frame, f"Score {sc:4.0f}   Align {result.alignment.score:3.0f}", (pad, pad + line * 2 - 6), 0.6 * scale, col, 2)
        _put(frame, f"Stability {result.stability:3.0f}   Symmetry {0 if not math.isfinite(result.symmetry) else result.symmetry:3.0f}",
             (pad, pad + line * 3 - 6), 0.5 * scale, WHITE, 1)
        hold = result.hold or {}
        status = "HOLDING" if hold.get("active") else "PAUSED"
        _put(frame, f"Hold {hold.get('held_s', 0):5.1f}/{hold.get('target_s', 0):.0f}s  {status}",
             (pad, pad + line * 4 - 6), 0.55 * scale, GREEN if hold.get("active") else AMBER, 1)
        _put(frame, f"Target: {result.target_pose}", (pad, pad + line * 5 - 6), 0.5 * scale, GREY, 1)

    # bottom message bar
    if result.message:
        bar_h = int(40 * scale)
        _panel(frame, 0, h - bar_h, w, h, 0.6)
        col = GREEN if result.is_correct else WHITE
        text = result.message if len(result.message) < 110 else result.message[:107] + "..."
        _put(frame, text, (pad, h - int(bar_h * 0.35)), 0.52 * scale, col, 1)
    return frame
