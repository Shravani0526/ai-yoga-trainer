"""Tests for the pipeline, personalisation, database, recurring errors and recommendations."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ai.feedback.recommendations import build_recommendations, session_feedback
from ai.pipeline import YogaPipeline
from ai.personalization.calibration import MAX_SHIFT_ANGLE, CalibrationRecorder, apply_calibration
from ai.personalization.difficulty import apply_difficulty, recommend_level
from ai.personalization.progress import overall_summary, per_pose_summary, practice_streak, prepare_sessions
from ai.personalization.recurring import detect_recurring_errors
from ai.pose.keypoints import KP
from ai.yoga.poses import criteria_for
from database.db import Database
from utils.drawing import annotate_frame
from utils.plotting import joint_error_figure, progress_figure, timeline_figure
from utils.synthetic import make_pose, to_pixels

FPS = 15


def run(pose, variant, coords, seconds=4, difficulty="Beginner", request="auto", noise=1.0, calibration=None, seed=1):
    rng = np.random.default_rng(seed)
    p = YogaPipeline(pose=pose, variant=request, difficulty=difficulty, calibration=calibration)
    p.start_session("synthetic")
    last = None
    for i in range(int(seconds * FPS)):
        last = p.analyze_keypoints(to_pixels(coords, noise_px=noise, rng=rng), i / FPS)
    return p, last


class PipelineTests(unittest.TestCase):
    def test_ideal_poses_run_the_full_pipeline(self):
        for name, variant in [("T Pose", "both"), ("Tree Pose", "right"), ("Warrior II", "left"),
                              ("Chair Pose", "both"), ("Triangle Pose", "left")]:
            with self.subTest(pose=name):
                p, r = run(name, variant, make_pose(name, variant))
                s = p.recorder.summary()
                self.assertTrue(r.is_correct)
                self.assertEqual(r.variant, variant)
                self.assertGreater(s["avg_score"], 80)
                self.assertGreater(s["held_s"], 2.5)
                self.assertEqual(r.corrections, [])

    def test_wrong_pose_does_not_run_the_hold_timer(self):
        p, r = run("Tree Pose", "auto", make_pose("T Pose"))
        self.assertFalse(r.is_correct)
        self.assertEqual(p.recorder.summary()["held_s"], 0.0)
        self.assertIn("T Pose", r.message)

    def test_flawed_pose_gives_corrections_and_no_hold(self):
        c = make_pose("Warrior II", "left")
        c[KP["left_ankle"]] = c[KP["left_knee"]] + np.array([0.45, 0.9])
        p, r = run("Warrior II", "left", c, difficulty="Intermediate")
        self.assertFalse(r.is_correct)
        self.assertEqual(p.recorder.summary()["held_s"], 0.0)
        self.assertTrue(r.corrections[0].text.startswith("Left knee (front leg):"))
        errors = {e["key"]: e for e in p.recorder.summary()["errors"]}
        self.assertGreater(errors["left_knee"]["bad_frac"], 0.9)
        self.assertEqual(errors["left_knee"]["direction"], "high")

    def test_auto_variant_picks_the_correct_side(self):
        _, r = run("Tree Pose", "auto", make_pose("Tree Pose", "right"))
        self.assertEqual(r.variant, "right")
        _, r = run("Triangle Pose", "auto", make_pose("Triangle Pose", "left"))
        self.assertEqual(r.variant, "left")

    def test_body_not_visible(self):
        p = YogaPipeline(pose="T Pose")
        k = to_pixels(make_pose("T Pose"))
        k[:, 2] = 0.05
        r = p.analyze_keypoints(k, 0.0)
        self.assertFalse(r.detected)
        self.assertIn("visible", r.message)

    def test_calibration_recording_roundtrip(self):
        p = YogaPipeline(pose="T Pose")
        rec = p.start_calibration()
        rng = np.random.default_rng(0)
        for i in range(30):
            p.analyze_keypoints(to_pixels(make_pose("T Pose"), noise_px=1.0, rng=rng), i / FPS)
        profile = rec.build_profile()
        self.assertEqual(profile["pose"], "T Pose")
        self.assertGreaterEqual(profile["n_samples"], 25)
        self.assertIn("left_knee", profile["metrics"])
        json.dumps(profile)  # must be JSON serialisable for the database


class PersonalisationTests(unittest.TestCase):
    def test_calibration_extends_but_is_capped(self):
        crit = criteria_for("T Pose", "both")
        rec = CalibrationRecorder("T Pose", "both")
        c = make_pose("T Pose")
        # user can only raise arms to ~60 degrees (limited mobility): far below the 80-110 target
        c[KP["left_wrist"]] = c[KP["left_elbow"]] + np.array([0.5, 0.4])
        rec.add({"left_shoulder": 65.0, "left_knee": 175.0})
        prof = rec.build_profile()
        out = {x.key: x for x in apply_calibration(crit, prof)}
        base = {x.key: x for x in crit}
        self.assertLess(out["left_shoulder"].low, base["left_shoulder"].low)
        self.assertGreaterEqual(out["left_shoulder"].low, base["left_shoulder"].low - MAX_SHIFT_ANGLE)
        self.assertEqual(out["left_shoulder"].high, base["left_shoulder"].high)
        self.assertEqual(out["left_knee"].low, base["left_knee"].low)  # inside range -> unchanged

    def test_calibration_only_applies_to_its_variant(self):
        rec = CalibrationRecorder("Tree Pose", "left")
        rec.add({"right_knee": 110.0, "left_knee": 170.0})
        prof = rec.build_profile()
        p = YogaPipeline(pose="Tree Pose", variant="auto", calibration={"left": prof})
        plain = YogaPipeline(pose="Tree Pose", variant="auto")
        got = {c.key: c for c in p.criteria_for_variant("right")}["right_knee"]
        ref = {c.key: c for c in plain.criteria_for_variant("right")}["right_knee"]
        self.assertEqual((got.low, got.high), (ref.low, ref.high))  # profile belongs to the left variant only
        got_l = {c.key: c for c in p.criteria_for_variant("left")}["right_knee"]
        ref_l = {c.key: c for c in plain.criteria_for_variant("left")}["right_knee"]
        self.assertGreater(got_l.high, ref_l.high)  # but it does apply to the left variant

    def test_difficulty_levels(self):
        crit = criteria_for("Warrior II", "left")
        easy = {c.key: c for c in apply_difficulty(crit, "Beginner")}
        hard = {c.key: c for c in apply_difficulty(crit, "Advanced")}
        base = {c.key: c for c in crit}
        self.assertLess(easy["left_knee"].low, base["left_knee"].low)
        self.assertGreater(easy["left_knee"].high, base["left_knee"].high)
        self.assertGreater(hard["left_knee"].low, base["left_knee"].low)
        self.assertLessEqual(hard["right_knee"].high, 180)

    def test_adaptive_level_rules(self):
        def hist(scores, done):
            return pd.DataFrame({"started_at": [f"2026-01-0{i + 1}" for i in range(len(scores))],
                                 "avg_score": scores, "completed": done})
        self.assertEqual(recommend_level(hist([85, 90], [1, 1]), "Beginner")[0], "Intermediate")
        self.assertEqual(recommend_level(hist([85, 90], [1, 0]), "Beginner")[0], "Beginner")
        self.assertEqual(recommend_level(hist([30, 40], [0, 0]), "Advanced")[0], "Intermediate")
        self.assertEqual(recommend_level(hist([90], [1]), "Beginner")[0], "Beginner")

    def test_streak(self):
        today = datetime(2026, 10, 6, tzinfo=timezone.utc).date()
        days = [today - timedelta(days=i) for i in (0, 1, 2, 5)]
        self.assertEqual(practice_streak(days, today), 3)
        self.assertEqual(practice_streak([today - timedelta(days=4)], today), 0)


def make_summary(pose, score, started, bad_knee=0.6, held=12.0, completed=0, direction="high"):
    return {
        "pose": pose, "variant": "left", "difficulty": "Beginner", "source": "test",
        "started_at": started, "duration_s": 30.0, "frames": 300, "detected_frames": 290,
        "correct_frames": 120, "held_s": held, "best_streak_s": held, "breaks": 2,
        "hold_target_s": 10.0, "completed": completed, "avg_score": score, "avg_alignment": score,
        "avg_stability": 70.0, "avg_symmetry": 80.0, "max_score": score + 5,
        "timeline": [{"t": 0.0, "score": score}],
        "errors": [{"key": "left_knee", "label": "Left knee (front leg)", "unit": "°", "frames_total": 290,
                    "frames_bad": int(290 * bad_knee), "bad_frac": bad_knee, "mean_deviation": 20.0,
                    "direction": direction, "mean_score": 50.0},
                   {"key": "torso_lean", "label": "Torso", "unit": "°", "frames_total": 290,
                    "frames_bad": 5, "bad_frac": 0.02, "mean_deviation": 3.0, "direction": "high", "mean_score": 97.0}],
    }


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database("sqlite:///:memory:")
        self.uid = self.db.get_or_create_user("Asha")

    def test_users_are_unique(self):
        self.assertEqual(self.uid, self.db.get_or_create_user("Asha"))
        self.assertNotEqual(self.uid, self.db.get_or_create_user("Ravi"))
        self.assertEqual(self.db.list_users(), ["Asha", "Ravi"])

    def test_session_roundtrip(self):
        sid = self.db.save_session(self.uid, make_summary("Warrior II", 70, "2026-10-01T08:00:00+00:00"))
        df = self.db.get_sessions(self.uid)
        self.assertEqual(len(df), 1)
        s = self.db.get_session(sid)
        self.assertEqual(s["pose"], "Warrior II")
        self.assertEqual(len(s["errors"]), 2)
        self.assertEqual(s["timeline"][0]["score"], 70)

    def test_calibration_and_levels(self):
        self.db.save_calibration(self.uid, "Tree Pose", "left", {"metrics": {"a": 1}})
        self.db.save_calibration(self.uid, "Tree Pose", "left", {"metrics": {"a": 2}})  # upsert
        self.assertEqual(self.db.get_calibrations(self.uid, "Tree Pose")["left"]["metrics"]["a"], 2)
        self.db.set_level(self.uid, "Tree Pose", "Beginner")
        self.db.set_level(self.uid, "Tree Pose", "Intermediate")
        self.assertEqual(self.db.get_levels(self.uid), {"Tree Pose": "Intermediate"})
        self.db.delete_calibrations(self.uid)
        self.assertEqual(self.db.get_calibrations(self.uid, "Tree Pose"), {})

    def test_backup_restore_skips_duplicates(self):
        self.db.save_session(self.uid, make_summary("T Pose", 80, "2026-10-01T08:00:00+00:00"))
        self.db.save_calibration(self.uid, "T Pose", "both", {"metrics": {}})
        backup = json.loads(json.dumps(self.db.export_user_json(self.uid)))
        other = Database("sqlite:///:memory:")
        uid2 = other.get_or_create_user("Copy")
        self.assertEqual(other.import_user_json(uid2, backup)["sessions_added"], 1)
        self.assertEqual(other.import_user_json(uid2, backup)["sessions_added"], 0)
        self.assertEqual(len(other.get_sessions(uid2)), 1)
        with self.assertRaises(ValueError):
            other.import_user_json(uid2, {"format": "nope"})

    def test_delete_user_removes_everything(self):
        self.db.save_session(self.uid, make_summary("T Pose", 80, "2026-10-01T08:00:00+00:00"))
        self.db.delete_user(self.uid)
        self.assertEqual(len(self.db.get_sessions(self.uid)), 0)
        self.assertEqual(len(self.db.get_errors(self.uid)), 0)

    def test_file_database_persists(self):
        with tempfile.TemporaryDirectory() as d:
            url = f"sqlite:///{os.path.join(d, 'sub', 'y.db')}"
            db1 = Database(url)
            u = db1.get_or_create_user("Maya")
            db1.save_session(u, make_summary("T Pose", 75, "2026-10-01T08:00:00+00:00"))
            db2 = Database(url)
            self.assertEqual(len(db2.get_sessions(db2.get_or_create_user("Maya"))), 1)


class RecurringAndRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.db = Database("sqlite:///:memory:")
        self.uid = self.db.get_or_create_user("Asha")
        base = datetime(2026, 9, 20, 8, tzinfo=timezone.utc)
        for i, score in enumerate([55, 58, 60, 62]):
            self.db.save_session(self.uid, make_summary("Warrior II", score, (base + timedelta(days=i)).isoformat()))

    def test_recurring_error_found_and_noise_ignored(self):
        rec = detect_recurring_errors(self.db.get_errors(self.uid))
        self.assertEqual(rec["key"].tolist(), ["left_knee"])
        row = rec.iloc[0]
        self.assertEqual(row["sessions_affected"], 4)
        self.assertEqual(row["direction"], "high")
        self.assertTrue(row["tip"])

    def test_no_recurring_error_with_single_session(self):
        one = self.db.get_errors(self.uid)
        one = one[one["session_id"] == one["session_id"].min()]
        self.assertTrue(detect_recurring_errors(one).empty)

    def test_recommendations(self):
        sessions = self.db.get_sessions(self.uid)
        rec = detect_recurring_errors(self.db.get_errors(self.uid))
        recs = build_recommendations(sessions, rec, {"Warrior II": "Beginner"},
                                     today=datetime(2026, 10, 6, tzinfo=timezone.utc))
        cats = [r.category for r in recs]
        self.assertIn("Focus", cats)
        self.assertIn("Next pose", cats)
        self.assertIn("Consistency", cats)
        self.assertEqual(recs[0].category, "Focus")
        self.assertTrue(all(r.reason for r in recs))

    def test_empty_history_gets_starter_recommendations(self):
        recs = build_recommendations(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(recs[0].category, "Getting started")

    def test_progress_tables(self):
        df = self.db.get_sessions(self.uid)
        s = overall_summary(df)
        self.assertEqual(s["sessions"], 4)
        self.assertAlmostEqual(s["best_score"], 62)
        table = per_pose_summary(df)
        self.assertEqual(table.loc[0, "Pose"], "Warrior II")
        self.assertGreater(table.loc[0, "Trend / session"], 0)

    def test_session_feedback_text(self):
        msgs = session_feedback(make_summary("Warrior II", 60, "2026-10-01T08:00:00+00:00"))
        self.assertTrue(any("Left knee" in m for m in msgs))


class DrawingAndPlotTests(unittest.TestCase):
    def test_annotate_and_plot(self):
        p, r = run("Warrior II", "left", make_pose("Warrior II", "left"), seconds=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for mirror in (False, True):
            out = annotate_frame(frame, r, mirror=mirror)
            self.assertEqual(out.shape, frame.shape)
            self.assertGreater(int(out.sum()), 0)
        s = p.recorder.summary()
        for fig in (timeline_figure(s["timeline"], 10), joint_error_figure(s["errors"]),
                    progress_figure(prepare_sessions(pd.DataFrame([make_summary("T Pose", 70, "2026-10-01T08:00:00+00:00")])))):
            self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
