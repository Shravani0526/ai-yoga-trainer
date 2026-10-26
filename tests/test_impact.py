"""Tests for the 'impact of a wrong pose' explanations."""

import unittest

from ai.feedback.impact import DISCLAIMER, EFFECTIVENESS, STRAIN, impact_for, impacts_for_corrections, session_impacts
from ai.yoga.poses import POSES
from utils.synthetic import make_pose


class ImpactCoverage(unittest.TestCase):
    def test_every_reachable_rule_has_an_impact(self):
        """Every joint rule that has a correction message must also explain its impact.

        Elbow/knee 'high' (> 180 degrees) cannot occur for a measured angle, so those are skipped.
        """
        missing = []
        for pose, spec in POSES.items():
            for variant, crits in spec.variants.items():
                for c in crits:
                    for direction, msg in (("low", c.too_low), ("high", c.too_high)):
                        if not msg:
                            continue
                        if c.is_angle and direction == "high" and c.high >= 180:
                            continue
                        if impact_for(pose, c.key, direction, c.label) is None:
                            missing.append((pose, variant, c.key, direction))
        self.assertEqual(missing, [])

    def test_kinds_are_valid_and_text_is_hedged(self):
        for pose, spec in POSES.items():
            for crits in spec.variants.values():
                for c in crits:
                    for d in ("low", "high"):
                        imp = impact_for(pose, c.key, d, c.label)
                        if imp:
                            self.assertIn(imp.kind, (STRAIN, EFFECTIVENESS))
                            self.assertTrue(imp.effect.endswith("."))
                            self.assertTrue(imp.long_term.endswith("."))
                            for banned in ("will cause", "guarantee", "cure", "injury is certain", "%"):
                                self.assertNotIn(banned, (imp.effect + imp.long_term).lower())

    def test_leg_roles_in_both_variants(self):
        # Tree Pose: the standing knee and the lifted knee have different meanings, on either side.
        for variant, standing, lifted in (("left", "left_knee", "right_knee"), ("right", "right_knee", "left_knee")):
            crits = {c.key: c for c in POSES["Tree Pose"].variants[variant]}
            self.assertEqual(impact_for("Tree Pose", standing, "low", crits[standing].label).area, "Standing knee")
            self.assertEqual(impact_for("Tree Pose", lifted, "high", crits[lifted].label).area, "Lifted knee")
        for variant, front in (("left", "left_knee"), ("right", "right_knee")):
            crits = {c.key: c for c in POSES["Warrior II"].variants[variant]}
            self.assertEqual(impact_for("Warrior II", front, "low", crits[front].label).area, "Front knee")

    def test_direction_matters(self):
        deep = impact_for("Chair Pose", "left_knee", "low", "Left knee")
        shallow = impact_for("Chair Pose", "left_knee", "high", "Left knee")
        self.assertEqual(deep.kind, STRAIN)
        self.assertEqual(shallow.kind, EFFECTIVENESS)

    def test_unknown_inputs_return_none(self):
        self.assertIsNone(impact_for("Tree Pose", "left_knee", "none"))
        self.assertIsNone(impact_for("Nonsense", "nothing", "low"))
        self.assertIn("not medical advice", DISCLAIMER)


class ImpactInSession(unittest.TestCase):
    def test_session_impacts_threshold_and_order(self):
        errors = [
            {"key": "left_knee", "label": "Left knee", "bad_frac": 0.10, "direction": "low"},
            {"key": "torso_lean", "label": "Torso", "bad_frac": 0.60, "direction": "high"},
            {"key": "ankle_gap", "label": "Stance width", "bad_frac": 0.40, "direction": "low"},
        ]
        rows = session_impacts("Warrior II", errors)
        self.assertEqual([r["joint"] for r in rows], ["Torso", "Stance width"])   # 10% row is below the threshold
        self.assertEqual(rows[0]["bad_pct"], 60)

    def test_pipeline_corrections_get_impacts(self):
        from tests.test_pipeline_db import run
        # a perfect pose needs no correction, so no impact is produced
        _, ok = run("Warrior II", "left", make_pose("Warrior II", "left"), seconds=3)
        self.assertEqual(impacts_for_corrections("Warrior II", ok.corrections), [])
        # a Triangle Pose skeleton judged against Warrior II is wrong: every correction should explain its impact
        p, bad = run("Warrior II", "left", make_pose("Triangle Pose", "left"), seconds=3)
        self.assertTrue(bad.corrections)
        imps = impacts_for_corrections("Warrior II", bad.corrections)
        self.assertEqual(len(imps), len(bad.corrections))
        rows = session_impacts("Warrior II", p.recorder.summary()["errors"])
        self.assertTrue(rows)
        self.assertTrue(all(r["effect"] and r["long_term"] for r in rows))


if __name__ == "__main__":
    unittest.main()
