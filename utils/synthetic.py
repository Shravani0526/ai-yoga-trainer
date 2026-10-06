"""Synthetic skeleton generator (forward kinematics) for tests, demos and Colab self-checks.

These are NOT recordings of people. They are idealised stick figures built from joint
directions, used only to verify that the rule engine behaves as designed (and to let the
app run a clearly-labelled demo when no camera / model is available).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ai.pose.keypoints import KP, NUM_KEYPOINTS

# Segment lengths in torso units.
THIGH, SHIN, UPPER_ARM, FOREARM = 1.0, 1.0, 0.65, 0.6
HIP_HALF, SHOULDER_HALF = 0.25, 0.45


def _dir(theta_deg: float) -> np.ndarray:
    """Unit vector rotated theta from 'straight down' towards +x (image right)."""
    t = np.radians(theta_deg)
    return np.array([np.sin(t), np.cos(t)])


def _build(spec: Dict[str, float], torso_lean: float = 0.0, torso_side: int = 1,
           width: float = 1.0) -> np.ndarray:
    """Create (17, 2) normalised coordinates (hip centre at origin, torso length 1).

    ``spec`` holds segment directions in degrees from straight-down (+x = image right,
    i.e. the subject's LEFT side when facing the camera).
    """
    p = np.zeros((NUM_KEYPOINTS, 2))
    hip_c = np.array([0.0, 0.0])
    up = np.array([np.sin(np.radians(torso_lean)) * torso_side, -np.cos(np.radians(torso_lean))])
    sh_c = hip_c + up
    across = np.array([-up[1], up[0]])           # perpendicular to the torso, points to +x when upright
    if across[0] < 0:
        across = -across

    p[KP["left_hip"]] = hip_c + np.array([HIP_HALF * width, 0.0])
    p[KP["right_hip"]] = hip_c - np.array([HIP_HALF * width, 0.0])
    p[KP["left_shoulder"]] = sh_c + SHOULDER_HALF * width * across
    p[KP["right_shoulder"]] = sh_c - SHOULDER_HALF * width * across
    p[KP["nose"]] = sh_c + up * 0.55
    p[KP["left_eye"]] = p[KP["nose"]] + np.array([0.05, -0.05])
    p[KP["right_eye"]] = p[KP["nose"]] + np.array([-0.05, -0.05])
    p[KP["left_ear"]] = p[KP["nose"]] + np.array([0.12, -0.02])
    p[KP["right_ear"]] = p[KP["nose"]] + np.array([-0.12, -0.02])

    for side, sgn in (("left", 1), ("right", -1)):
        hip = p[KP[f"{side}_hip"]]
        knee = hip + THIGH * _dir(spec[f"{side}_thigh"])
        ankle = knee + SHIN * _dir(spec[f"{side}_shin"])
        p[KP[f"{side}_knee"]], p[KP[f"{side}_ankle"]] = knee, ankle
        sh = p[KP[f"{side}_shoulder"]]
        elbow = sh + UPPER_ARM * _dir(spec[f"{side}_arm"])
        wrist = elbow + FOREARM * _dir(spec[f"{side}_forearm"])
        p[KP[f"{side}_elbow"]], p[KP[f"{side}_wrist"]] = elbow, wrist
    return p


def _base() -> Dict[str, float]:
    return {"left_thigh": 0, "left_shin": 0, "right_thigh": 0, "right_shin": 0,
            "left_arm": 0, "left_forearm": 0, "right_arm": 0, "right_forearm": 0}


def _t_pose() -> np.ndarray:
    s = _base()
    s.update(left_arm=90, left_forearm=90, right_arm=-90, right_forearm=-90)
    return _build(s)


def _tree(stand: str) -> np.ndarray:
    s = _base()
    lift = "right" if stand == "left" else "left"
    sgn = 1 if lift == "left" else -1
    # Lifted thigh opens outwards, shin comes back so the foot rests on the inner thigh.
    s[f"{lift}_thigh"] = 60 * sgn
    s[f"{lift}_shin"] = -55 * sgn
    # Hands in prayer position in front of the chest (elbows out, wrists together).
    s.update(left_arm=55, left_forearm=-120, right_arm=-55, right_forearm=120)
    p = _build(s)
    # Prayer: pull both wrists to the chest centre.
    chest = (p[KP["left_shoulder"]] + p[KP["right_shoulder"]]) / 2 + np.array([0.0, 0.35])
    p[KP["left_wrist"]] = chest + np.array([0.04, 0.0])
    p[KP["right_wrist"]] = chest - np.array([0.04, 0.0])
    return p


def _warrior(front: str) -> np.ndarray:
    s = _base()
    back = "right" if front == "left" else "left"
    fs = 1 if front == "left" else -1
    s[f"{front}_thigh"] = 70 * fs     # thigh close to horizontal, pointing outwards
    s[f"{front}_shin"] = 0            # shin vertical
    s[f"{back}_thigh"] = -35 * fs
    s[f"{back}_shin"] = -35 * fs
    s.update(left_arm=90, left_forearm=90, right_arm=-90, right_forearm=-90)
    return _build(s)


def _chair() -> np.ndarray:
    # Side view: person faces image right. Both legs/arms coincide (tiny offsets keep them distinct).
    s = _base()
    for side, off in (("left", 2), ("right", -2)):
        s[f"{side}_thigh"] = 62 + off
        s[f"{side}_shin"] = -12 + off
        s[f"{side}_arm"] = 180 - 30 + off      # reaching up and slightly forward
        s[f"{side}_forearm"] = 180 - 30 + off
    return _build(s, torso_lean=28, torso_side=1, width=0.15)


def _triangle(down: str) -> np.ndarray:
    s = _base()
    ds = 1 if down == "left" else -1
    s["left_thigh"], s["left_shin"] = 30, 30
    s["right_thigh"], s["right_shin"] = -30, -30
    s[f"{down}_arm"], s[f"{down}_forearm"] = 0, 0                      # hanging down towards the shin
    up_side = "right" if down == "left" else "left"
    s[f"{up_side}_arm"], s[f"{up_side}_forearm"] = 180, 180            # straight up
    return _build(s, torso_lean=60, torso_side=ds)


def _standing() -> np.ndarray:
    """Relaxed standing (arms by the sides) - used to check that nothing is mis-classified."""
    return _build(_base())


def make_pose(name: str, variant: str = "both") -> np.ndarray:
    """Return ideal normalised coordinates (17, 2) for a pose (or "Standing")."""
    if name == "Standing":
        return _standing()
    if name == "T Pose":
        return _t_pose()
    if name == "Tree Pose":
        return _tree(variant if variant in ("left", "right") else "left")
    if name == "Warrior II":
        return _warrior(variant if variant in ("left", "right") else "left")
    if name == "Chair Pose":
        return _chair()
    if name == "Triangle Pose":
        return _triangle(variant if variant in ("left", "right") else "left")
    raise ValueError(f"No synthetic generator for pose '{name}'")


def to_pixels(coords: np.ndarray, width: int = 640, height: int = 480, torso_px: float = 110.0,
              center: Optional[tuple] = None, conf: float = 0.9,
              noise_px: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Convert normalised coordinates into MoveNet-style pixel keypoints (17, 3)."""
    if center is None:
        center = (width / 2.0, height * 0.42)
    kps = np.zeros((NUM_KEYPOINTS, 3))
    kps[:, :2] = coords * torso_px + np.array(center)
    if noise_px > 0:
        rng = rng or np.random.default_rng(0)
        kps[:, :2] += rng.normal(0.0, noise_px, size=(NUM_KEYPOINTS, 2))
    kps[:, 2] = conf
    return kps


def perturb(coords: np.ndarray, joint: str, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Move one keypoint (to simulate an alignment error)."""
    out = coords.copy()
    out[KP[joint]] += np.array([dx, dy])
    return out
