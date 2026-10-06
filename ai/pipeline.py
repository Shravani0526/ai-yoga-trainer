"""End-to-end pipeline used by BOTH the Colab notebook and the Streamlit app.

Camera/Video -> MoveNet -> 17 keypoints -> normalisation -> smoothing -> joint angles ->
pose classification -> alignment analysis -> explainable correction -> scoring ->
stability -> hold timer -> session history (recorder) -> personalisation inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ai.feedback.corrections import Correction, generate_corrections
from ai.personalization.calibration import CalibrationRecorder, apply_calibration
from ai.personalization.difficulty import LevelConfig, apply_difficulty, get_level
from ai.pose.geometry import compute_metrics
from ai.pose.keypoints import NUM_KEYPOINTS
from ai.pose.normalization import normalize_keypoints
from ai.pose.smoothing import KeypointSmoother
from ai.yoga.alignment import AlignmentResult, evaluate_alignment
from ai.yoga.classifier import MIN_CLASS_SCORE, NO_POSE, ClassificationResult, classify
from ai.yoga.criteria import Criterion
from ai.yoga.hold_timer import HoldTimer
from ai.yoga.poses import criteria_for, get_pose, resolve_variants
from ai.yoga.scoring import ScoreBreakdown, compute_overall
from ai.yoga.session import SessionRecorder
from ai.yoga.stability import StabilityTracker
from ai.yoga.symmetry import symmetry_score

VARIANT_SWITCH_MARGIN = 10.0
MIN_VISIBLE_WEIGHT = 0.6
MIN_KEY_JOINT_SCORE = 40.0   # no important joint (weight >= 1) may be badly off for the hold to count


@dataclass
class FrameResult:
    t: float
    detected: bool
    message: str
    keypoints_display: Optional[np.ndarray] = None       # (17, 3) pixels, smoothed
    detected_pose: str = NO_POSE                           # what the classifier thinks (any pose)
    detected_variant: Optional[str] = None
    class_score: float = 0.0                               # classifier score of the TARGET pose
    target_pose: str = ""
    variant: Optional[str] = None                          # variant used for alignment
    alignment: Optional[AlignmentResult] = None
    corrections: List[Correction] = field(default_factory=list)
    stability: float = float("nan")
    symmetry: float = float("nan")
    breakdown: Optional[ScoreBreakdown] = None
    is_correct: bool = False
    hold: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    visible_fraction: float = 0.0

    @property
    def overall(self) -> float:
        return self.breakdown.overall if self.breakdown else float("nan")


class YogaPipeline:
    """Stateful per-stream pipeline (one instance per camera stream / video)."""

    def __init__(self, detector=None, pose: str = "T Pose", variant: str = "auto",
                 difficulty: str = "Beginner", calibration: Optional[Dict[str, Dict]] = None,
                 min_conf: float = 0.25, source: str = "webcam",
                 hold_target_s: Optional[float] = None):
        self.detector = detector
        self.min_conf = min_conf
        self.source = source
        self.smoother = KeypointSmoother()
        self._center_s: Optional[np.ndarray] = None
        self._torso_s: Optional[float] = None
        self.stability = StabilityTracker()
        self.timer = HoldTimer()
        self.recorder: Optional[SessionRecorder] = None
        self.calibrator: Optional[CalibrationRecorder] = None
        self._variant_used: Optional[str] = None
        self.configure(pose, variant, difficulty, calibration, hold_target_s)

    # ------------------------------------------------------------------ configuration
    def configure(self, pose: str, variant: str = "auto", difficulty: str = "Beginner",
                  calibration: Optional[Dict[str, Dict]] = None, hold_target_s: Optional[float] = None) -> None:
        """(Re)configure the target pose; resets all temporal state.

        ``calibration`` maps variant name -> calibration profile. A profile is only ever applied
        to the variant it was recorded for (a left-leg Tree Pose says nothing about the right leg).
        """
        self.spec = get_pose(pose)
        self.pose = pose
        self.variant_request = variant or "auto"
        self.difficulty = difficulty
        self.level: LevelConfig = get_level(difficulty)
        self.calibration = calibration
        self.hold_target_s = float(hold_target_s) if hold_target_s else self.level.hold_target_s
        self._criteria: Dict[str, List[Criterion]] = {}
        for v in self.spec.variant_names:
            base = criteria_for(pose, v)
            self._criteria[v] = apply_difficulty(apply_calibration(base, (calibration or {}).get(v)), difficulty)
        self.timer = HoldTimer(target_s=self.hold_target_s, strict=self.level.strict_hold)
        self.reset_state()

    def reset_state(self) -> None:
        self.smoother.reset()
        self.stability.reset()
        self.timer.reset()
        self._center_s, self._torso_s = None, None
        self._variant_used = None
        if self.recorder is not None:
            self.start_session(self.source)

    def start_session(self, source: Optional[str] = None) -> SessionRecorder:
        if source:
            self.source = source
        self.recorder = SessionRecorder(self.pose, self.variant_request, self.difficulty,
                                        self.source, self.hold_target_s)
        self.timer.reset()
        return self.recorder

    def start_calibration(self) -> CalibrationRecorder:
        self.calibrator = CalibrationRecorder(self.pose, self.variant_request)
        return self.calibrator

    def criteria_for_variant(self, variant: str) -> List[Criterion]:
        return list(self._criteria[variant])

    # ------------------------------------------------------------------ processing
    def process_frame(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        if self.detector is None:
            raise RuntimeError("No detector configured - pass a MoveNetDetector or call analyze_keypoints().")
        return self.analyze_keypoints(self.detector.detect(frame_bgr), t)

    def analyze_keypoints(self, kps_px: np.ndarray, t: float) -> FrameResult:
        res = FrameResult(t=t, detected=False, message="", target_pose=self.pose)
        norm = normalize_keypoints(kps_px, self.min_conf)
        if norm is None:
            res.message = "Step back so your shoulders, hips and legs are fully visible."
            res.hold = self._hold_dict(self.timer.update(t, False))
            self._record(res)
            return res

        # --- smoothing (body-centred coordinates) and display coordinates
        coords = self.smoother(norm.coords, t)
        if self._center_s is None:
            self._center_s, self._torso_s = norm.center_px.copy(), norm.torso_px
        else:
            self._center_s = 0.5 * self._center_s + 0.5 * norm.center_px
            self._torso_s = 0.5 * self._torso_s + 0.5 * norm.torso_px
        display = np.zeros((NUM_KEYPOINTS, 3))
        display[:, :2] = np.where(np.isnan(coords), 0.0, coords * self._torso_s + self._center_s)
        display[:, 2] = np.where(np.isnan(coords).any(axis=1), 0.0, norm.confidence)
        res.keypoints_display = display
        res.visible_fraction = norm.visible_fraction

        # --- joint angles / distances
        metrics = compute_metrics(coords)
        res.metrics = metrics

        # --- pose classification (all poses, transparent rules)
        cls = classify(metrics)
        res.detected_pose, res.detected_variant = cls.pose, cls.variant
        variant = self._choose_variant(cls)
        res.variant = variant
        res.class_score = cls.score_for(self.pose, variant)

        # --- alignment against the personalised, level-adjusted rules
        alignment = evaluate_alignment(metrics, self._criteria[variant])
        res.alignment = alignment
        res.corrections = generate_corrections(alignment.joint_results)

        # --- stability + symmetry + overall score
        res.stability = self.stability.update(t, coords, self._center_s / self._torso_s)
        res.symmetry, _ = symmetry_score(metrics)
        res.breakdown = compute_overall(alignment.score, res.stability, res.symmetry, self.spec.symmetric)

        # --- correctness gate for the hold timer
        pose_ok = res.class_score >= MIN_CLASS_SCORE
        visible_ok = alignment.visible_weight_fraction >= MIN_VISIBLE_WEIGHT
        key_scores = [j.score for j in alignment.joint_results
                      if j.weight >= 1.0 and math.isfinite(j.score)]
        align_ok = (math.isfinite(alignment.score) and alignment.score >= self.level.min_alignment
                    and (not key_scores or min(key_scores) >= MIN_KEY_JOINT_SCORE))
        stable_ok = res.stability >= self.level.min_stability
        res.is_correct = bool(pose_ok and visible_ok and align_ok and stable_ok)
        res.detected = True
        res.hold = self._hold_dict(self.timer.update(t, res.is_correct))
        res.message = self._status_message(res, pose_ok, visible_ok, align_ok, stable_ok)

        if self.calibrator is not None:
            self.calibrator.add(metrics, coords)
        self._record(res)
        return res

    # ------------------------------------------------------------------ helpers
    def _choose_variant(self, cls: ClassificationResult) -> str:
        options = resolve_variants(self.pose, self.variant_request)
        if len(options) == 1:
            self._variant_used = options[0]
            return options[0]
        scores = {v: cls.score_for(self.pose, v) for v in options}
        best = max(scores, key=scores.get)
        if self._variant_used is None or scores[best] > scores.get(self._variant_used, 0.0) + VARIANT_SWITCH_MARGIN:
            self._variant_used = best
        return self._variant_used

    @staticmethod
    def _hold_dict(state) -> Dict[str, Any]:
        return {"held_s": state.held_s, "streak_s": state.streak_s, "best_streak_s": state.best_streak_s,
                "active": state.active, "breaks": state.breaks, "target_s": state.target_s,
                "completed": state.completed, "progress": state.progress}

    def _status_message(self, res: FrameResult, pose_ok: bool, visible_ok: bool,
                        align_ok: bool, stable_ok: bool) -> str:
        if not visible_ok:
            return "Some joints are hidden - make sure your whole body is in view."
        if not pose_ok:
            if res.detected_pose not in (NO_POSE, self.pose):
                return f"This looks like {res.detected_pose}. Move into {self.pose}."
            return f"Move into {self.pose} - see the instructions on the Pose Selection page."
        if res.corrections:
            return res.corrections[0].text
        if not stable_ok:
            return "Good alignment - hold still to start the timer."
        if res.is_correct:
            return "Great posture - keep holding steady."
        return "Almost there - fine-tune your alignment."

    def _record(self, res: FrameResult) -> None:
        if self.recorder is None:
            return
        self.recorder.add_frame(
            t=res.t, detected=res.detected, alignment=res.alignment,
            overall=res.overall, stability=res.stability, symmetry=res.symmetry,
            is_correct=res.is_correct, hold=res.hold or self._hold_dict(self.timer.state),
            variant_used=res.variant,
            correction_keys=[c.key for c in res.corrections],
        )
