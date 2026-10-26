"""Pose library: transparent geometric rules for five yoga poses.

IMPORTANT: these are hand-written *geometric target ranges*, not a trained model. They are
based on the commonly taught alignment of each pose, are deliberately editable, and are
refined per user through calibration (see ai.personalization.calibration).

Camera placement matters because joint angles are measured in 2-D image space:
    * T Pose, Tree Pose, Warrior II, Triangle Pose -> face the camera (front view)
    * Chair Pose -> stand side-on to the camera (knee / hip bend is visible from the side)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai.pose.keypoints import KP
from ai.yoga.criteria import Criterion, mirror_criteria

TORSO = "x torso"


def _both(*criteria: Criterion) -> List[Criterion]:
    """Return each criterion plus its left/right mirror."""
    out: List[Criterion] = []
    for c in criteria:
        out.append(c)
    out.extend(mirror_criteria(criteria))
    return out


@dataclass(frozen=True)
class PoseSpec:
    name: str
    sanskrit: str
    description: str
    instructions: Tuple[str, ...]
    camera_hint: str
    symmetric: bool
    variants: Dict[str, Tuple[Criterion, ...]]
    variant_labels: Dict[str, str]
    benefits: str = ""

    @property
    def variant_names(self) -> List[str]:
        return list(self.variants.keys())

    @property
    def is_sided(self) -> bool:
        return set(self.variants.keys()) == {"left", "right"}


# --------------------------------------------------------------------------- T POSE
def _t_pose() -> PoseSpec:
    ls, le, lw = KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]
    lh, lk, la = KP["left_hip"], KP["left_knee"], KP["left_ankle"]
    arm = _both(
        Criterion("left_shoulder", "Left shoulder", "left_shoulder", 80, 110, 30, 1.5,
                  "Raise your left arm to shoulder height", "Lower your left arm to shoulder height",
                  joints=(ls,), signature=True, gate=True,
                  tip="Practise arm raises against a wall: back of the hands touch the wall at shoulder height."),
        Criterion("left_elbow", "Left elbow", "left_elbow", 160, 180, 30, 1.0,
                  "Straighten your left arm", "Bend your left elbow slightly",
                  joints=(le,), signature=True,
                  tip="Reach out through the fingertips and keep the elbow soft but straight."),
    )
    legs = _both(
        Criterion("left_knee", "Left knee", "left_knee", 165, 180, 30, 1.0,
                  "Straighten your left knee", "Soften your left knee",
                  joints=(lk,), signature=True, gate=True,
                  tip="Engage the thigh muscles and press evenly through the foot."),
        Criterion("left_hip", "Left hip", "left_hip", 160, 180, 30, 0.5,
                  "Stand taller through your left hip", "Stack your left hip under your shoulder",
                  joints=(lh,),
                  tip="Imagine a line from the ear through shoulder and hip to the ankle."),
    )
    body = [
        Criterion("torso_lean", "Torso", "torso_lean", 0, 8, 15, 1.0,
                  "", "Stand upright - keep your torso vertical", joints=(ls, KP["right_shoulder"]),
                  tip="Lengthen the spine; avoid leaning to either side."),
        Criterion("shoulder_tilt", "Shoulder line", "shoulder_tilt", 0, 6, 12, 0.8,
                  "", "Level your shoulders", joints=(ls, KP["right_shoulder"]),
                  tip="Practise in front of a mirror and check both shoulders are the same height."),
        Criterion("ankle_gap", "Feet distance", "ankle_gap", 0.0, 1.1, 0.6, 0.6,
                  "", "Bring your feet closer together", unit=TORSO, joints=(la, KP["right_ankle"]),
                  signature=True, gate=True,
                  tip="Stand with the feet hip-width apart or together."),
    ]
    return PoseSpec(
        name="T Pose",
        sanskrit="Tadasana with arms out",
        description="Standing upright with both arms stretched out sideways at shoulder height.",
        instructions=(
            "Stand tall, feet together or hip-width apart, facing the camera.",
            "Raise both arms out to the sides until they are level with your shoulders.",
            "Straighten elbows and knees, keep shoulders and hips level.",
        ),
        camera_hint="Front view - stand facing the camera with your whole body visible.",
        symmetric=True,
        variants={"both": tuple(arm + legs + body)},
        variant_labels={"both": "Both sides"},
        benefits="Builds shoulder endurance and trains upright posture awareness.",
    )


# --------------------------------------------------------------------------- TREE POSE
def _tree_pose() -> PoseSpec:
    # "left" variant = standing on the LEFT leg, RIGHT foot lifted. "right" is mirrored.
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lk, rk = KP["left_knee"], KP["right_knee"]
    la, ra = KP["left_ankle"], KP["right_ankle"]
    left = [
        Criterion("left_knee", "Left knee (standing leg)", "left_knee", 165, 180, 30, 1.5,
                  "Straighten your standing left leg", "Soften your left knee",
                  joints=(lk,), signature=True, gate=True,
                  tip="Press the standing foot into the floor and lift the kneecap."),
        Criterion("right_knee", "Right knee (lifted leg)", "right_knee", 20, 90, 35, 1.2,
                  "Open your right knee a little", "Bend your right knee more and draw the foot up",
                  joints=(rk,), signature=True, gate=True,
                  tip="Use your hand to guide the foot to the calf or inner thigh - never the knee joint."),
        Criterion("right_foot_lift", "Right foot height", "right_foot_lift", 0.45, 1.6, 0.5, 1.2,
                  "Lift your right foot higher (calf or inner thigh, not the knee)",
                  "Lower your right foot a little", unit=TORSO, joints=(ra,),
                  signature=True, gate=True,
                  tip="Start with the foot on the ankle or calf and raise it gradually as balance improves."),
        Criterion("wrist_gap", "Hands distance", "wrist_gap", 0.0, 0.5, 0.5, 0.8,
                  "", "Bring your palms together at your chest", unit=TORSO,
                  joints=(KP["left_wrist"], KP["right_wrist"]), signature=True,
                  tip="Press the palms together at the heart centre to engage the chest."),
        Criterion("wrist_level", "Hands height", "wrist_level", 0.3, 1.2, 0.5, 0.5,
                  "Raise your hands to your chest", "Lower your hands to your chest", unit=TORSO,
                  joints=(KP["left_wrist"], KP["right_wrist"]),
                  tip="Keep the thumbs resting lightly against the breastbone."),
        Criterion("torso_lean", "Torso", "torso_lean", 0, 10, 15, 1.0,
                  "", "Stand upright - keep your torso vertical", joints=(ls, rs),
                  tip="Fix your gaze on a still point and lengthen the spine to stop swaying."),
        Criterion("hip_tilt", "Hip line", "hip_tilt", 0, 8, 15, 0.8,
                  "", "Level your hips", joints=(KP["left_hip"], KP["right_hip"]),
                  tip="Keep both hip bones facing forward; do not hike the lifted hip."),
        Criterion("shoulder_tilt", "Shoulder line", "shoulder_tilt", 0, 8, 12, 0.6,
                  "", "Level your shoulders", joints=(ls, rs),
                  tip="Relax the shoulders away from the ears."),
    ]
    return PoseSpec(
        name="Tree Pose",
        sanskrit="Vrksasana",
        description="Balance on one leg with the other foot resting on the calf or inner thigh, palms together at the chest.",
        instructions=(
            "Face the camera and shift your weight onto one leg.",
            "Place the other foot on your calf or inner thigh (not on the knee).",
            "Bring your palms together at the chest and fix your gaze on a still point.",
            "Choose 'Auto' to let the trainer detect which leg you stand on.",
        ),
        camera_hint="Front view - the camera must see both ankles and both hands.",
        symmetric=False,
        variants={"left": tuple(left), "right": tuple(mirror_criteria(left))},
        variant_labels={"left": "Standing on left leg", "right": "Standing on right leg"},
        benefits="Improves balance, ankle and hip stability and concentration.",
    )


# --------------------------------------------------------------------------- WARRIOR II
def _warrior_ii() -> PoseSpec:
    # "left" variant = LEFT leg is the front (bent) leg, RIGHT leg extended behind.
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lk, rk = KP["left_knee"], KP["right_knee"]
    left = [
        Criterion("left_knee", "Left knee (front leg)", "left_knee", 85, 115, 35, 1.6,
                  "Straighten your left knee a little", "Bend your left knee deeper",
                  joints=(lk,), signature=True, gate=True,
                  tip="Sink the hips until the thigh approaches parallel with the floor; keep the knee over the ankle."),
        Criterion("right_knee", "Right knee (back leg)", "right_knee", 165, 180, 30, 1.4,
                  "Straighten your back right leg", "Soften your right knee",
                  joints=(rk,), signature=True, gate=True,
                  tip="Press the outer edge of the back foot down and straighten the leg strongly."),
        Criterion("ankle_gap", "Stance width", "ankle_gap", 1.5, 3.5, 0.8, 1.0,
                  "Widen your stance", "Narrow your stance a little", unit=TORSO,
                  joints=(KP["left_ankle"], KP["right_ankle"]), signature=True, gate=True,
                  tip="Step the feet roughly one leg-length apart."),
        Criterion("left_shoulder", "Left arm height", "left_shoulder", 80, 110, 30, 1.0,
                  "Raise your left arm to shoulder height", "Lower your left arm to shoulder height",
                  joints=(ls,), signature=True,
                  tip="Imagine your arms resting on a table at shoulder height."),
        Criterion("right_shoulder", "Right arm height", "right_shoulder", 80, 110, 30, 1.0,
                  "Raise your right arm to shoulder height", "Lower your right arm to shoulder height",
                  joints=(rs,), signature=True,
                  tip="Imagine your arms resting on a table at shoulder height."),
        Criterion("left_elbow", "Left elbow", "left_elbow", 160, 180, 30, 0.7,
                  "Straighten your left arm", "Bend your left elbow slightly",
                  joints=(KP["left_elbow"],), tip="Stretch actively through the fingertips."),
        Criterion("right_elbow", "Right elbow", "right_elbow", 160, 180, 30, 0.7,
                  "Straighten your right arm", "Bend your right elbow slightly",
                  joints=(KP["right_elbow"],), tip="Stretch actively through the fingertips."),
        Criterion("torso_lean", "Torso", "torso_lean", 0, 10, 15, 1.0,
                  "", "Stack your torso upright over your hips", joints=(ls, rs),
                  tip="Lengthen the spine and keep the chest centred between the feet."),
        Criterion("shoulder_tilt", "Shoulder line", "shoulder_tilt", 0, 8, 12, 0.6,
                  "", "Level your shoulders", joints=(ls, rs),
                  tip="Relax both shoulders down away from the ears."),
    ]
    return PoseSpec(
        name="Warrior II",
        sanskrit="Virabhadrasana II",
        description="Wide stance, front knee bent over the ankle, back leg straight, arms extended at shoulder height.",
        instructions=(
            "Step your feet about one leg-length apart, facing the camera.",
            "Bend the front knee towards 90 degrees and keep the back leg straight.",
            "Extend both arms at shoulder height and keep the torso upright.",
            "Choose 'Auto' to let the trainer detect which leg is in front.",
        ),
        camera_hint="Front view - stand far enough back that both feet and both hands are visible.",
        symmetric=False,
        variants={"left": tuple(left), "right": tuple(mirror_criteria(left))},
        variant_labels={"left": "Left leg in front", "right": "Right leg in front"},
        benefits="Strengthens legs and shoulders, opens the hips and builds stamina.",
    )


# --------------------------------------------------------------------------- CHAIR POSE
def _chair_pose() -> PoseSpec:
    ls = KP["left_shoulder"]
    side = _both(
        Criterion("left_knee", "Left knee", "left_knee", 80, 125, 35, 1.5,
                  "Lift your hips slightly", "Bend your left knee more - sit back into the chair",
                  joints=(KP["left_knee"],), signature=True, gate=True,
                  tip="Imagine sitting back onto a low chair; keep weight in the heels."),
        Criterion("left_hip", "Left hip", "left_hip", 70, 130, 35, 1.0,
                  "Lift your chest and open your left hip angle", "Sit your hips back and down",
                  joints=(KP["left_hip"],), signature=True,
                  tip="Hinge from the hips and keep the spine long."),
        Criterion("left_shoulder", "Left arm raise", "left_shoulder", 140, 180, 35, 1.0,
                  "Reach your left arm up beside your ear", "Lower your left arm slightly",
                  joints=(ls,), signature=True,
                  tip="Reach up through the fingertips and keep the shoulders relaxed."),
        Criterion("left_elbow", "Left elbow", "left_elbow", 155, 180, 30, 0.7,
                  "Straighten your left arm", "Soften your left elbow",
                  joints=(KP["left_elbow"],), tip="Stretch actively through the fingertips."),
    )
    body = [
        Criterion("torso_lean", "Torso", "torso_lean", 0, 40, 20, 0.8,
                  "", "Lift your chest - keep your spine long", joints=(ls, KP["right_shoulder"]),
                  tip="Draw the ribs in and lengthen the front of the body."),
        Criterion("ankle_gap", "Feet distance", "ankle_gap", 0.0, 1.1, 0.6, 0.5,
                  "", "Bring your feet closer together", unit=TORSO,
                  joints=(KP["left_ankle"], KP["right_ankle"]), signature=True,
                  tip="Keep the big toes touching or hip-width apart."),
    ]
    return PoseSpec(
        name="Chair Pose",
        sanskrit="Utkatasana",
        description="Knees bent as if sitting back into a chair, spine long, arms reaching overhead.",
        instructions=(
            "Stand side-on to the camera with your whole body in the frame.",
            "Bend both knees and sit your hips back and down.",
            "Reach both arms up beside your ears and keep the chest lifted.",
        ),
        camera_hint="Side view - stand side-on to the camera so the knee and hip bend is visible.",
        symmetric=True,
        variants={"both": tuple(side + body)},
        variant_labels={"both": "Both sides"},
        benefits="Strengthens thighs, glutes and core and builds leg endurance.",
    )


# --------------------------------------------------------------------------- TRIANGLE
def _triangle_pose() -> PoseSpec:
    # "left" variant = LEFT hand reaches down (torso bends towards the left leg).
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    left = [
        Criterion("left_knee", "Left knee", "left_knee", 165, 180, 30, 1.3,
                  "Straighten your left leg", "Soften your left knee",
                  joints=(KP["left_knee"],), signature=True, gate=True,
                  tip="Lift the kneecaps and keep both legs strong."),
        Criterion("right_knee", "Right knee", "right_knee", 165, 180, 30, 1.3,
                  "Straighten your right leg", "Soften your right knee",
                  joints=(KP["right_knee"],), signature=True, gate=True,
                  tip="Lift the kneecaps and keep both legs strong."),
        Criterion("ankle_gap", "Stance width", "ankle_gap", 1.4, 3.4, 0.8, 1.0,
                  "Widen your stance", "Narrow your stance a little", unit=TORSO,
                  joints=(KP["left_ankle"], KP["right_ankle"]), signature=True, gate=True,
                  tip="Step the feet about one leg-length apart."),
        Criterion("torso_lean", "Side bend", "torso_lean", 30, 85, 25, 1.4,
                  "Bend further sideways, reaching long", "Lift your torso slightly - do not collapse",
                  joints=(ls, rs), signature=True, gate=True,
                  tip="Reach out through the front hand first, then tilt the torso as one long line."),
        Criterion("left_wrist_ankle", "Lower hand to shin", "left_wrist_ankle", 0.0, 1.0, 0.8, 1.2,
                  "", "Reach your left hand further down towards your shin", unit=TORSO,
                  joints=(KP["left_wrist"], KP["left_ankle"]), signature=True,
                  tip="Rest the lower hand on the shin or a block - never on the knee joint."),
        Criterion("wrist_line_vertical", "Arms in one line", "wrist_line_vertical", 0, 20, 25, 1.2,
                  "", "Stack your arms in one vertical line",
                  joints=(KP["left_wrist"], KP["right_wrist"]), signature=True,
                  tip="Imagine your arms are on a pane of glass; open the chest towards the ceiling."),
        Criterion("left_elbow", "Left elbow", "left_elbow", 160, 180, 30, 0.7,
                  "Straighten your left arm", "Soften your left elbow",
                  joints=(KP["left_elbow"],), tip="Keep both arms long and active."),
        Criterion("right_elbow", "Right elbow", "right_elbow", 160, 180, 30, 0.7,
                  "Straighten your right arm", "Soften your right elbow",
                  joints=(KP["right_elbow"],), tip="Keep both arms long and active."),
    ]
    return PoseSpec(
        name="Triangle Pose",
        sanskrit="Trikonasana",
        description="Wide straight-legged stance, torso bent sideways over one leg, bottom hand to the shin and top arm reaching up.",
        instructions=(
            "Step your feet about one leg-length apart, facing the camera, legs straight.",
            "Reach one arm out and tilt the torso sideways over that leg.",
            "Rest the lower hand on your shin or ankle and reach the other arm straight up.",
            "Choose 'Auto' to let the trainer detect which side you bend towards.",
        ),
        camera_hint="Front view - keep both feet, both hands and the head inside the frame.",
        symmetric=False,
        variants={"left": tuple(left), "right": tuple(mirror_criteria(left))},
        variant_labels={"left": "Left hand down", "right": "Right hand down"},
        benefits="Stretches hamstrings, hips and spine and strengthens the legs.",
    )


POSES: Dict[str, PoseSpec] = {
    spec.name: spec
    for spec in (_t_pose(), _tree_pose(), _warrior_ii(), _chair_pose(), _triangle_pose())
}

POSE_ORDER = ["T Pose", "Tree Pose", "Warrior II", "Chair Pose", "Triangle Pose"]


def list_poses() -> List[str]:
    return list(POSE_ORDER)


def get_pose(name: str) -> PoseSpec:
    try:
        return POSES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pose '{name}'. Available: {list_poses()}") from exc


def criteria_for(name: str, variant: str) -> List[Criterion]:
    spec = get_pose(name)
    if variant not in spec.variants:
        raise ValueError(f"Pose '{name}' has no variant '{variant}' (choose from {spec.variant_names})")
    return list(spec.variants[variant])


def find_criterion(pose_name: str, key: str) -> Optional[Criterion]:
    """Look a criterion up by key in any variant of a pose (used for tips)."""
    if pose_name not in POSES:
        return None
    for crit_list in POSES[pose_name].variants.values():
        for c in crit_list:
            if c.key == key:
                return c
    return None


def resolve_variants(pose_name: str, variant: str) -> List[str]:
    """'auto' -> all variants of the pose; otherwise just the requested one."""
    spec = get_pose(pose_name)
    if variant in (None, "", "auto"):
        return spec.variant_names
    if variant not in spec.variants:
        return spec.variant_names[:1]
    return [variant]
