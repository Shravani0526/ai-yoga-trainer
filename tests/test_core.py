"""Unit tests for the geometry, normalisation, smoothing, classifier, corrections and timers.

Run with:  python -m unittest discover -s tests -t . -v      (or: pytest -q)

All tests use synthetic stick-figure skeletons from utils/synthetic.py. They verify that the
rule engine behaves as designed; they say nothing about MoveNet's accuracy on real video.
"""

import math
import unittest

import numpy as np

from ai.feedback.corrections import generate_corrections
from ai.pose.geometry import angle_3pts, compute_metrics, tilt_from_horizontal, tilt_from_vertical
from ai.pose.keypoints import KP
from ai.pose.normalization import normalize_keypoints
from ai.pose.smoothing import KeypointSmoother
from ai.yoga.alignment import evaluate_alignment
from ai.yoga.classifier import NO_POSE, classify
from ai.yoga.criteria import Criterion, mirror_criterion, score_value, swap_sides
from ai.yoga.hold_timer import HoldTimer
from ai.yoga.poses import POSES, criteria_for, list_poses
from ai.yoga.scoring import compute_overall
from ai.yoga.stability import StabilityTracker
from ai.yoga.symmetry import symmetry_score
from utils.synthetic import make_pose, to_pixels

CASES = [("T Pose", "both"), ("Tree Pose", "left"), ("Tree Pose", "right"),
         ("Warrior II", "left"), ("Warrior II", "right"), ("Chair Pose", "both"),
         ("Triangle Pose", "left"), ("Triangle Pose", "right")]


class GeometryTests(unittest.TestCase):
    def test_angle(self):
        self.assertAlmostEqual(angle_3pts([0, 1], [0, 0], [1, 0]), 90.0)
        self.assertAlmostEqual(angle_3pts([-1, 0], [0, 0], [1, 0]), 180.0)
        self.assertTrue(math.isnan(angle_3pts([np.nan, 0], [0, 0], [1, 0])))

    def test_tilts(self):
        self.assertAlmostEqual(tilt_from_horizontal([0, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(tilt_from_horizontal([0, 0], [-1, 1]), 45.0)
        self.assertAlmostEqual(tilt_from_vertical([0, 0], [0, 5]), 0.0)

    def test_t_pose_metrics(self):
        m = compute_metrics(make_pose("T Pose"))
        self.assertAlmostEqual(m["left_elbow"], 180.0, places=3)
        self.assertAlmostEqual(m["left_knee"], 180.0, places=3)
        self.assertAlmostEqual(m["torso_lean"], 0.0, places=3)
        self.assertTrue(80 <= m["left_shoulder"] <= 110)

    def test_missing_keypoint_gives_nan_not_guess(self):
        c = make_pose("T Pose")
        c[KP["left_knee"]] = np.nan
        m = compute_metrics(c)
        self.assertTrue(math.isnan(m["left_knee"]))
        self.assertFalse(math.isnan(m["right_knee"]))


class NormalizationTests(unittest.TestCase):
    def test_scale_and_translation_invariance(self):
        c = make_pose("Warrior II", "left")
        a = normalize_keypoints(to_pixels(c, torso_px=100, center=(300, 200)))
        b = normalize_keypoints(to_pixels(c, torso_px=210, center=(500, 260)))
        np.testing.assert_allclose(a.coords, b.coords, atol=1e-9)
        self.assertAlmostEqual(a.torso_px, 100.0, places=6)

    def test_returns_none_without_torso(self):
        k = to_pixels(make_pose("T Pose"))
        k[KP["left_hip"], 2] = 0.05
        self.assertIsNone(normalize_keypoints(k))

    def test_low_confidence_points_become_nan(self):
        k = to_pixels(make_pose("T Pose"))
        k[KP["left_wrist"], 2] = 0.1
        n = normalize_keypoints(k)
        self.assertTrue(np.isnan(n.coords[KP["left_wrist"]]).all())


class SmoothingTests(unittest.TestCase):
    def test_reduces_jitter(self):
        rng = np.random.default_rng(0)
        base = make_pose("T Pose")
        sm = KeypointSmoother()
        raw_err, sm_err = [], []
        for i in range(120):
            noisy = base + rng.normal(0, 0.03, base.shape)
            out = sm(noisy, i / 15.0)
            if i > 30:
                raw_err.append(np.abs(noisy - base).mean())
                sm_err.append(np.abs(out - base).mean())
        self.assertLess(np.mean(sm_err), 0.7 * np.mean(raw_err))

    def test_follows_real_motion(self):
        sm = KeypointSmoother()
        c = make_pose("T Pose")
        for i in range(20):
            out = sm(c, i / 15.0)
        moved = c + np.array([0.5, 0.0])
        for i in range(20, 35):
            out = sm(moved, i / 15.0)
        self.assertLess(np.abs(out - moved).max(), 0.05)

    def test_missing_points_are_held_briefly_then_dropped(self):
        sm = KeypointSmoother(max_gap_s=0.3)
        c = make_pose("T Pose")
        sm(c, 0.0)
        gap = c.copy()
        gap[KP["left_wrist"]] = np.nan
        held = sm(gap, 0.1)
        self.assertFalse(np.isnan(held[KP["left_wrist"]]).any())
        for i in range(2, 10):
            out = sm(gap, i * 0.1)
        self.assertTrue(np.isnan(out[KP["left_wrist"]]).all())


class ClassifierTests(unittest.TestCase):
    def test_ideal_poses_are_recognised(self):
        for name, variant in CASES:
            with self.subTest(pose=name, variant=variant):
                r = classify(compute_metrics(make_pose(name, variant)))
                self.assertEqual(r.pose, name)
                if variant != "both":
                    self.assertEqual(r.variant, variant)
                self.assertGreater(r.margin, 10)

    def test_standing_is_not_a_pose(self):
        r = classify(compute_metrics(make_pose("Standing")))
        self.assertEqual(r.pose, NO_POSE)

    def test_survives_pixel_noise(self):
        rng = np.random.default_rng(3)
        for name, variant in CASES:
            hits = 0
            for _ in range(20):
                n = normalize_keypoints(to_pixels(make_pose(name, variant), noise_px=1.5, rng=rng))
                hits += classify(compute_metrics(n.coords)).pose == name
            self.assertGreaterEqual(hits, 18, f"{name} {variant}: {hits}/20")


class AlignmentAndCorrectionTests(unittest.TestCase):
    def test_ideal_pose_has_full_alignment(self):
        for name, variant in CASES:
            al = evaluate_alignment(compute_metrics(make_pose(name, variant)), criteria_for(name, variant))
            self.assertGreaterEqual(al.score, 99.0, f"{name} {variant}")
            self.assertEqual(generate_corrections(al.joint_results), [])

    def test_score_value_shape(self):
        self.assertEqual(score_value(100, 90, 110, 30), 100.0)
        self.assertAlmostEqual(score_value(125, 90, 110, 30), 50.0)
        self.assertEqual(score_value(200, 90, 110, 30), 0.0)

    def test_documented_example_sentence(self):
        c = Criterion("right_knee", "Right knee", "right_knee", 90, 110, tol=100,
                      too_low="Straighten your right knee", too_high="Bend your right knee")
        al = evaluate_alignment({"right_knee": 132.0}, [c])
        corr = generate_corrections(al.joint_results)[0]
        self.assertEqual(corr.text, "Right knee: 132°. Target: 90–110°. Bend your right knee slightly.")
        self.assertAlmostEqual(corr.deviation, 22.0)

    def test_every_rule_points_the_right_way(self):
        """Out-of-range values must produce an instruction of the matching direction."""
        low_words = {"knee": ("straighten", "lift your hips", "open"), "elbow": ("straighten",),
                     "shoulder": ("raise", "reach")}
        high_words = {"knee": ("bend", "soften"), "elbow": ("bend", "soften"), "shoulder": ("lower",)}
        checked = 0
        for name in list_poses():
            for variant, crits in POSES[name].variants.items():
                for c in crits:
                    kind = c.metric.split("_")[-1]
                    if kind not in low_words or c.key != c.metric:
                        continue
                    if c.low > 0 and c.too_low:
                        text = c.too_low.lower()
                        self.assertTrue(any(w in text for w in low_words[kind]), f"{name}/{variant}/{c.key}: '{c.too_low}'")
                        checked += 1
                    if c.high < 180 and c.too_high:
                        text = c.too_high.lower()
                        self.assertTrue(any(w in text for w in high_words[kind]), f"{name}/{variant}/{c.key}: '{c.too_high}'")
                        checked += 1
        self.assertGreater(checked, 20)

    def test_flawed_warrior_gets_the_right_correction(self):
        c = make_pose("Warrior II", "left")
        c[KP["left_ankle"]] = c[KP["left_knee"]] + np.array([0.45, 0.9])   # front knee too straight
        al = evaluate_alignment(compute_metrics(c), criteria_for("Warrior II", "left"))
        corr = generate_corrections(al.joint_results)
        self.assertEqual(corr[0].key, "left_knee")
        self.assertEqual(corr[0].direction, "high")
        self.assertIn("Bend your left knee", corr[0].text)

    def test_mirroring(self):
        self.assertEqual(swap_sides("Stand upright with your left knee"), "Stand upright with your right knee")
        c = criteria_for("Tree Pose", "left")[1]
        m = mirror_criterion(c)
        self.assertEqual((c.key, m.key), ("right_knee", "left_knee"))
        self.assertEqual(m.joints, (KP["left_knee"],))


class TimerAndStabilityTests(unittest.TestCase):
    def test_timer_counts_only_while_correct(self):
        t = HoldTimer(target_s=3.0, enter_s=0.2, grace_s=0.3)
        now = 0.0
        for _ in range(30):              # 3 s correct @10 fps
            now += 0.1
            t.update(now, True)
        counted = t.state.held_s
        self.assertGreater(counted, 2.5)
        for _ in range(30):              # 3 s incorrect: nothing may be added
            now += 0.1
            t.update(now, False)
        self.assertAlmostEqual(t.state.held_s, counted, places=6)
        self.assertEqual(t.state.breaks, 1)
        self.assertFalse(t.state.active)

    def test_grace_period_keeps_hold_alive(self):
        t = HoldTimer(target_s=10, enter_s=0.1, grace_s=0.5)
        now = 0.0
        for _ in range(10):
            now += 0.1
            t.update(now, True)
        for _ in range(3):
            now += 0.1
            t.update(now, False)         # 0.3 s glitch < grace
        now += 0.1
        t.update(now, True)
        self.assertEqual(t.state.breaks, 0)

    def test_big_frame_gaps_are_clipped(self):
        t = HoldTimer(target_s=10, enter_s=0.0)
        t.update(0.0, True)
        t.update(30.0, True)             # a 30 s stall must not add 30 s
        self.assertLessEqual(t.state.held_s, 0.25 + 1e-9)

    def test_completion_and_strict_mode(self):
        loose, strict = HoldTimer(target_s=2.0, enter_s=0.0, grace_s=0.1), HoldTimer(target_s=2.0, enter_s=0.0, grace_s=0.1, strict=True)
        now = 0.0
        pattern = ([True] * 10 + [False] * 5) * 3          # 1 s on / 0.5 s off
        for ok in pattern:
            now += 0.1
            loose.update(now, ok)
            strict.update(now, ok)
        self.assertTrue(loose.state.completed)
        self.assertFalse(strict.state.completed)

    def test_stability_distinguishes_still_from_moving(self):
        base = make_pose("T Pose")
        still, moving = StabilityTracker(), StabilityTracker()
        rng = np.random.default_rng(1)
        for i in range(40):
            tt = i / 15.0
            s = still.update(tt, base + rng.normal(0, 0.003, base.shape), np.array([3.0, 2.0]))
            wobble = base + np.array([0.15 * math.sin(tt * 6), 0.1 * math.cos(tt * 6)])
            m = moving.update(tt, wobble, np.array([3.0 + 0.3 * math.sin(tt * 6), 2.0]))
        self.assertGreater(s, 85)
        self.assertLess(m, 40)


class ScoringTests(unittest.TestCase):
    def test_weights(self):
        b = compute_overall(100, 50, 100, symmetric_pose=True)
        self.assertAlmostEqual(b.overall, 0.65 * 100 + 0.20 * 50 + 0.15 * 100)
        b2 = compute_overall(80, 60, 10, symmetric_pose=False)
        self.assertAlmostEqual(b2.overall, 0.75 * 80 + 0.25 * 60)

    def test_symmetry_detects_asymmetry(self):
        sym, _ = symmetry_score(compute_metrics(make_pose("T Pose")))
        c = make_pose("T Pose")
        c[KP["left_wrist"]] = c[KP["left_elbow"]] + np.array([0.3, -0.5])
        asym, _ = symmetry_score(compute_metrics(c))
        self.assertGreater(sym, 95)
        self.assertLess(asym, sym - 5)


if __name__ == "__main__":
    unittest.main()
