"""MoveNet keypoint constants (COCO 17-point layout)."""

from __future__ import annotations

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

KP = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Bones used for drawing the skeleton overlay.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # face
    (0, 5), (0, 6),                            # neck
    (5, 6), (5, 11), (6, 12), (11, 12),        # torso
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# Body joints (everything except face) used by stability / normalisation.
BODY_INDICES = list(range(5, 17))
# Joints needed to define the torso frame used for normalisation.
TORSO_INDICES = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]


def is_left(idx: int) -> bool:
    """True for keypoints on the subject's left side (odd COCO indices)."""
    return idx > 0 and idx % 2 == 1


def mirror_index(idx: int) -> int:
    """Return the keypoint index of the opposite body side (nose maps to itself)."""
    if idx == 0:
        return 0
    return idx + 1 if idx % 2 == 1 else idx - 1
