"""Impact of a wrong pose: WHY a correction matters.

For every joint rule that can be broken, this module holds a short, plain-language explanation of
what that mistake does to the pose (its *effect*) and what it can lead to if it is repeated (the
*possible long-term effect*). Each entry is tagged as one of two kinds:

* ``strain``         - the mistake can put extra load on a joint, the spine or a muscle group.
* ``effectiveness``  - the mistake mainly makes the pose less useful (muscles do not work, balance is lost).

Honest scope: these are general yoga-teaching explanations written once by the project authors, not
measurements from the user's body and not medical advice. They are worded with "can" / "may" because
one imperfect repetition is rarely harmful - the risk comes from holding or repeating a fault for a long time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

STRAIN = "strain"
EFFECTIVENESS = "effectiveness"

DISCLAIMER = ("General yoga-teaching information, not medical advice. One imperfect attempt is normal; the effects "
              "described here relate to holding or repeating the same fault for a long time. If you feel pain, "
              "come out of the pose and ask a qualified instructor or doctor.")


@dataclass(frozen=True)
class Impact:
    kind: str          # STRAIN or EFFECTIVENESS
    area: str          # body area affected, e.g. "Knee"
    effect: str        # what the mistake does to the pose / body right now (one sentence)
    long_term: str     # what repeating it can lead to (one sentence)

    @property
    def live_text(self) -> str:
        return f"{self.area}: {self.effect}"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "area": self.area, "effect": self.effect, "long_term": self.long_term}


def _i(kind: str, area: str, effect: str, long_term: str) -> Impact:
    return Impact(kind, area, effect, long_term)


# ---- pose-specific entries: (pose, base key, direction) ------------------------------------------------
# "low" = measured value is BELOW the target range, "high" = ABOVE it.  Side prefixes are stripped, so
# "left_knee" and "right_knee" both look up "knee"; where the two legs have different jobs the base key
# is looked up with a role suffix first (see _role_keys).
_POSE: Dict[tuple, Impact] = {
    # ---- Tree Pose
    ("Tree Pose", "stand_knee", "low"): _i(STRAIN, "Standing knee",
        "A bent standing knee makes the leg wobble and the thigh muscles stop holding you up.",
        "Balancing on a soft, bent knee again and again loads the knee joint and the kneecap and trains weak balance habits."),
    ("Tree Pose", "lift_knee", "high"): _i(EFFECTIVENESS, "Lifted knee",
        "With the knee not opened out, the foot stays low and the hip does not open.",
        "Hip stiffness is not reduced and the pose stays at a beginner level."),
    ("Tree Pose", "lift_knee", "low"): _i(STRAIN, "Lifted knee",
        "Pulling the knee in too tightly twists the knee rather than rotating the hip.",
        "Forcing the foot up by twisting the knee, instead of rotating from the hip, can irritate the knee ligaments."),
    ("Tree Pose", "foot_lift", "high"): _i(STRAIN, "Knee joint",
        "A foot placed at or above the knee presses sideways on the standing knee joint.",
        "Sideways pressure on the knee joint is the classic Tree Pose mistake and can strain the knee ligaments over time; keep the foot on the calf or inner thigh."),
    ("Tree Pose", "foot_lift", "low"): _i(EFFECTIVENESS, "Hip",
        "A very low foot gives less hip opening and less of a balance challenge.",
        "Progress is slower, but this is a safe place to start - raise the foot gradually."),
    ("Tree Pose", "wrist_gap", "high"): _i(EFFECTIVENESS, "Chest and shoulders",
        "Hands drifting apart lose the pressing action at the chest that helps balance.",
        "Balance cues are weaker and the chest and shoulder muscles are less engaged."),
    ("Tree Pose", "wrist_level", "low"): _i(EFFECTIVENESS, "Shoulders",
        "Hands that sink low let the shoulders round forward.",
        "Rounded shoulders held often reinforce a slouched upper back."),
    ("Tree Pose", "wrist_level", "high"): _i(EFFECTIVENESS, "Neck and shoulders",
        "Hands drifting up tense the shoulders towards the ears.",
        "Constant shoulder shrugging builds neck and upper-trapezius tension."),
    ("Tree Pose", "torso_lean", "high"): _i(EFFECTIVENESS, "Balance and spine",
        "Leaning to one side moves your centre of gravity off the standing foot, so balance is lost.",
        "Repeated side-leaning makes the muscles on one side of the spine do more work than the other."),
    ("Tree Pose", "hip_tilt", "high"): _i(STRAIN, "Lower back and hip",
        "A hiked or dropped hip tilts the pelvis, so the standing hip and lower back take uneven load.",
        "Holding an uneven pelvis repeatedly can tire and strain the lower back and the standing hip."),

    # ---- Warrior II
    ("Warrior II", "front_knee", "low"): _i(STRAIN, "Front knee",
        "Sinking deeper than a right angle pushes the knee forward past the ankle and loads it heavily.",
        "Repeated deep knee bends with the knee past the toes put extra stress on the knee joint and kneecap."),
    ("Warrior II", "front_knee", "high"): _i(EFFECTIVENESS, "Thighs and hips",
        "A shallow front knee means the thigh muscles do much less work.",
        "Leg strength and stamina build more slowly, and the hips are not opened."),
    ("Warrior II", "back_knee", "low"): _i(EFFECTIVENESS, "Back leg",
        "A bent back leg stops it acting as an anchor, so the front knee takes more of your weight.",
        "The front knee ends up doing most of the work, and the inner-thigh stretch is lost."),
    ("Warrior II", "ankle_gap", "low"): _i(STRAIN, "Front knee",
        "With the feet too close, the knee can only bend to a right angle by travelling past the ankle.",
        "A narrow stance repeatedly forces the front knee forward over the toes."),
    ("Warrior II", "ankle_gap", "high"): _i(STRAIN, "Inner thighs and groin",
        "A stance that is too wide over-stretches the inner thighs and makes the pose hard to hold.",
        "Over-stretching the groin muscles can lead to inner-thigh strain."),
    ("Warrior II", "shoulder", "low"): _i(EFFECTIVENESS, "Shoulders",
        "Arms drooping below shoulder height let the shoulder muscles switch off.",
        "Shoulder endurance is not built and the chest closes."),
    ("Warrior II", "shoulder", "high"): _i(EFFECTIVENESS, "Neck and shoulders",
        "Arms raised above shoulder height shrug the shoulders up and tighten the neck.",
        "Repeated shrugging builds neck and shoulder tension."),
    ("Warrior II", "elbow", "low"): _i(EFFECTIVENESS, "Arms",
        "Bent elbows shorten the arm line and reduce the work done by the shoulders.",
        "The shoulders gain less endurance."),
    ("Warrior II", "torso_lean", "high"): _i(STRAIN, "Lower back",
        "Leaning towards the front leg moves your weight onto the front knee and bends the spine sideways.",
        "Repeated leaning over the front leg loads the front knee and the lower back."),
    ("Warrior II", "shoulder_tilt", "high"): _i(EFFECTIVENESS, "Neck and shoulders",
        "Uneven shoulders show one side of the neck and shoulder is working harder.",
        "One-sided tension in the neck and upper back."),

    # ---- Chair Pose
    ("Chair Pose", "knee", "low"): _i(STRAIN, "Knees",
        "Sitting deeper than the target range closes the knee angle and loads the knees and kneecaps.",
        "Deep, repeated knee bends under load can irritate the kneecap, especially if the knees travel over the toes."),
    ("Chair Pose", "knee", "high"): _i(EFFECTIVENESS, "Thighs and glutes",
        "A shallow squat means the thighs and glutes do little work.",
        "Leg endurance builds slowly."),
    ("Chair Pose", "hip", "low"): _i(STRAIN, "Lower back",
        "A hip angle that is too folded pushes the chest forward and rounds or overloads the lower back.",
        "Holding a folded-forward Chair Pose repeatedly strains the lower back."),
    ("Chair Pose", "hip", "high"): _i(EFFECTIVENESS, "Hips and thighs",
        "Staying too upright means you are not sitting back, so the glutes do not work.",
        "The pose becomes a half-squat that trains the thighs less."),
    ("Chair Pose", "shoulder", "low"): _i(EFFECTIVENESS, "Shoulders and upper back",
        "Arms that do not reach overhead let the chest drop and the upper back round.",
        "The shoulder-opening and posture benefit is lost."),
    ("Chair Pose", "elbow", "low"): _i(EFFECTIVENESS, "Arms",
        "Bent elbows shorten the arm line.", "Less shoulder and upper-back work."),
    ("Chair Pose", "torso_lean", "high"): _i(STRAIN, "Lower back",
        "Leaning the trunk far forward shifts weight to the toes and the lower back.",
        "A habitually forward-folded Chair Pose overworks the lower back."),
    ("Chair Pose", "ankle_gap", "high"): _i(EFFECTIVENESS, "Balance and thighs",
        "Feet too far apart turn the pose into a wide squat.", "Balance and inner-thigh work change; it is a different exercise."),

    # ---- Triangle Pose
    ("Triangle Pose", "knee", "low"): _i(EFFECTIVENESS, "Back of the legs",
        "A bent knee takes the stretch off the hamstrings.",
        "Hamstring flexibility improves more slowly, and the weight shifts onto the bent knee."),
    ("Triangle Pose", "ankle_gap", "low"): _i(EFFECTIVENESS, "Balance and legs",
        "A narrow stance is unstable and limits how far you can bend sideways.",
        "The side stretch is shallow and balance is poor."),
    ("Triangle Pose", "ankle_gap", "high"): _i(STRAIN, "Inner thighs and hamstrings",
        "An over-wide stance over-stretches the inner thighs and hamstrings.",
        "Repeated over-stretching can strain the groin and hamstring muscles."),
    ("Triangle Pose", "torso_lean", "low"): _i(EFFECTIVENESS, "Side body",
        "Not bending sideways enough means the waist and side-body stretch is missed.",
        "Spinal side-flexibility is not trained."),
    ("Triangle Pose", "torso_lean", "high"): _i(STRAIN, "Lower back and shoulder",
        "Collapsing deeper than the range dumps weight on the lower hand and compresses the side of the waist.",
        "Repeatedly collapsing into the bottom arm strains the lower back and the supporting shoulder."),
    ("Triangle Pose", "wrist_ankle", "high"): _i(EFFECTIVENESS, "Spine and hamstrings",
        "A hand that does not reach the shin shows you are bending less than the range or rounding the back to reach.",
        "If you round the spine to reach lower, the lower back takes strain instead of the hamstrings and side body."),
    ("Triangle Pose", "wrist_line_vertical", "high"): _i(EFFECTIVENESS, "Chest and shoulders",
        "Arms not stacked in one line close the chest and put one shoulder in front.",
        "Shoulders and chest are not opened; the neck may tense as you look up."),
    ("Triangle Pose", "elbow", "low"): _i(EFFECTIVENESS, "Arms", "Bent elbows shorten the arm line.", "Less shoulder work and a smaller chest opening."),

    # ---- T Pose
    ("T Pose", "shoulder", "low"): _i(EFFECTIVENESS, "Shoulders",
        "Arms below shoulder height stop the shoulder muscles working.", "Shoulder endurance is not trained."),
    ("T Pose", "shoulder", "high"): _i(EFFECTIVENESS, "Neck and shoulders",
        "Arms above shoulder height lift the shoulders towards the ears.", "Repeated shrugging builds neck tension."),
    ("T Pose", "knee", "low"): _i(EFFECTIVENESS, "Legs",
        "Soft or bent knees mean the thighs are not engaged and the hips sit back.", "Upright-posture awareness is weaker."),
    ("T Pose", "hip", "low"): _i(EFFECTIVENESS, "Hips and spine",
        "A hip angle that is not straight shows you are slouching or tilting the pelvis.", "Slouching habits are reinforced."),
    ("T Pose", "ankle_gap", "high"): _i(EFFECTIVENESS, "Balance",
        "Feet apart wider than hip width reduce the upright-posture practice.", "It becomes a wide stance rather than a tall one."),
}

# ---- generic fall-backs by joint family (used when no pose-specific entry exists) -----------------------
_GENERIC: Dict[tuple, Impact] = {
    ("knee", "low"): _i(STRAIN, "Knee", "A knee bent more than the target loads the knee joint.", "Repeating this can irritate the knee."),
    ("knee", "high"): _i(EFFECTIVENESS, "Legs", "A straighter knee than the target means the thighs do less work.", "Leg strength builds more slowly."),
    ("hip", "low"): _i(STRAIN, "Lower back", "A hip angle below the range overloads the lower back.", "Repeated overload can tire the lower back."),
    ("hip", "high"): _i(EFFECTIVENESS, "Hips", "The hip is not bending as the pose asks.", "The hips are not worked or opened."),
    ("shoulder", "low"): _i(EFFECTIVENESS, "Shoulders", "The arm is lower than the pose needs.", "Shoulder muscles are not trained."),
    ("shoulder", "high"): _i(EFFECTIVENESS, "Shoulders", "The arm is higher than the pose needs.", "Shoulders and neck tense up."),
    ("elbow", "low"): _i(EFFECTIVENESS, "Arms", "Bent elbows shorten the arm line.", "The arms do less work."),
    ("torso_lean", "high"): _i(STRAIN, "Spine", "Leaning more than the range puts uneven load on one side of the spine.", "Repeated leaning overworks one side of the back."),
    ("shoulder_tilt", "high"): _i(EFFECTIVENESS, "Neck and shoulders", "Uneven shoulders mean one side is working harder.", "One-sided neck and shoulder tension."),
    ("hip_tilt", "high"): _i(STRAIN, "Lower back", "An uneven pelvis shifts load onto one side of the lower back.", "Repeating this can strain the lower back."),
    ("ankle_gap", "low"): _i(EFFECTIVENESS, "Stability", "The feet are closer than the pose needs.", "Stability and depth are limited."),
    ("ankle_gap", "high"): _i(STRAIN, "Inner thighs", "The feet are wider than the pose needs.", "Over-stretching the inner thighs."),
}

_SIDES = ("left_", "right_")


def _strip_side(key: str) -> str:
    for s in _SIDES:
        if key.startswith(s):
            return key[len(s):]
    return key


def _lookup_keys(pose: str, key: str, label: str = "") -> List[str]:
    """Candidate base keys, most specific first (a role key distinguishes the two legs of Tree/Warrior II)."""
    base = _strip_side(key)
    lab = label.lower()
    keys: List[str] = []
    if base == "knee":
        if pose == "Tree Pose":
            keys.append("stand_knee" if "standing" in lab else "lift_knee")
        elif pose == "Warrior II":
            keys.append("front_knee" if "front" in lab else "back_knee")
    if base == "right_foot_lift" or base.endswith("foot_lift"):
        keys.append("foot_lift")
    if base.endswith("wrist_ankle"):
        keys.append("wrist_ankle")
    keys.append(base)
    return keys


def impact_for(pose: str, key: str, direction: str, label: str = "") -> Optional[Impact]:
    """Impact of the joint rule ``key`` being out of range in ``direction`` ('low' or 'high'), or None."""
    if direction not in ("low", "high"):
        return None
    for base in _lookup_keys(pose, key, label):
        hit = _POSE.get((pose, base, direction))
        if hit:
            return hit
    for base in _lookup_keys(pose, key, label):
        hit = _GENERIC.get((base, direction))
        if hit:
            return hit
    return None


def impacts_for_corrections(pose: str, corrections: Sequence) -> List[Impact]:
    """One Impact per correction that has one (duplicates removed, order kept)."""
    out: List[Impact] = []
    seen = set()
    for c in corrections:
        imp = impact_for(pose, c.key, c.direction, c.label)
        if imp and imp not in seen:
            seen.add(imp)
            out.append(imp)
    return out


def session_impacts(pose: str, summary_errors: Sequence[Dict], min_frac: float = 0.25, max_items: int = 4) -> List[Dict]:
    """Impacts for the joints that were out of range for a meaningful part of a session.

    ``summary_errors`` is ``SessionRecorder.summary()['errors']``.  Returns dicts ready for display, worst first.
    """
    rows: List[Dict] = []
    seen = set()
    for e in sorted(summary_errors, key=lambda r: r.get("bad_frac", 0.0), reverse=True):
        if e.get("bad_frac", 0.0) < min_frac:
            continue
        imp = impact_for(pose, e["key"], e.get("direction", "none"), e.get("label", ""))
        if not imp or imp in seen:
            continue
        seen.add(imp)
        rows.append({"joint": e.get("label", e["key"]), "bad_pct": round(100 * e["bad_frac"]), **imp.as_dict()})
        if len(rows) == max_items:
            break
    return rows
