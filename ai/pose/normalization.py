"""Keypoint normalisation: translation + scale invariant body coordinates.

Pixel keypoints are converted into a body-centred frame:
    * origin  = mid-hip
    * unit    = torso length (mid-shoulder to mid-hip)
    * axes    = image orientation (x right, y down); no rotation is applied, because the
                rules need to know whether the torso is leaning or the arms are level.

Aspect ratio is preserved, so joint angles computed from normalised coordinates are
identical to the angles measured in the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ai.pose.keypoints import KP, NUM_KEYPOINTS

MIN_TORSO_PIXELS = 20.0


@dataclass
class NormalizedPose:
    coords: np.ndarray        # (17, 2) body-centred, torso-scaled; NaN where not visible
    confidence: np.ndarray    # (17,)
    center_px: np.ndarray     # (2,) mid-hip in pixels
    torso_px: float           # torso length in pixels
    visible_fraction: float   # fraction of the 12 body keypoints above the threshold


def normalize_keypoints(kps_px: np.ndarray, min_conf: float = 0.25) -> Optional[NormalizedPose]:
    """Normalise ``kps_px`` (17, 3) = (x, y, score) in pixels.

    Returns ``None`` if the torso cannot be located reliably (shoulders / hips missing).
    """
    kps = np.asarray(kps_px, dtype=float)
    if kps.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected keypoints of shape (17, 3), got {kps.shape}")

    conf = kps[:, 2]
    xy = kps[:, :2].copy()
    ok = conf >= min_conf

    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    if not (ok[ls] and ok[rs] and ok[lh] and ok[rh]):
        return None

    mid_sh = (xy[ls] + xy[rs]) / 2.0
    mid_hip = (xy[lh] + xy[rh]) / 2.0
    torso = float(np.linalg.norm(mid_sh - mid_hip))
    if torso < MIN_TORSO_PIXELS:
        return None

    coords = (xy - mid_hip) / torso
    coords[~ok] = np.nan
    visible = float(np.mean(ok[5:17]))
    return NormalizedPose(coords=coords, confidence=conf.copy(), center_px=mid_hip,
                          torso_px=torso, visible_fraction=visible)


def denormalize(coords: np.ndarray, center_px: np.ndarray, torso_px: float) -> np.ndarray:
    """Inverse of :func:`normalize_keypoints` for the xy part (useful for plotting)."""
    return np.asarray(coords, dtype=float) * torso_px + np.asarray(center_px, dtype=float)
