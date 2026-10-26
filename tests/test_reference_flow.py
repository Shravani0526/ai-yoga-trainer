"""Tests for the reference illustrations and the guided-flow defaults used by app.py."""

import unittest

import numpy as np

from ai.pipeline import YogaPipeline
from ai.yoga.poses import get_pose, list_poses, resolve_variants
from utils.reference import render_reference, reference_caption
from utils.synthetic import make_pose, to_pixels


class ReferenceImageTests(unittest.TestCase):
    def test_every_pose_and_variant_renders(self):
        for name in list_poses():
            spec = get_pose(name)
            for v in spec.variant_names:
                img = render_reference(name, v, mirror=True)
                self.assertEqual(img.shape, (460, 360, 3))
                self.assertGreater(len(np.unique(img.reshape(-1, 3), axis=0)), 5, f"{name}/{v} looks blank")

    def test_mirror_flips_the_picture(self):
        a = render_reference("Warrior II", "left", mirror=False, show_targets=False)
        b = render_reference("Warrior II", "left", mirror=True, show_targets=False)
        self.assertFalse(np.array_equal(a, b))
        # the warrior's front (left) foot is on the right of the unmirrored picture and on the left of the mirrored one
        def foot_x(img):
            ys, xs = np.where((img[:, :, 0] < 60) & (img[:, :, 1] > 130) & (img[:, :, 2] > 120) & (img[:, :, 2] < 160))
            return xs.mean()
        self.assertLess(foot_x(b), foot_x(a))

    def test_sizes_and_target_toggle(self):
        small = render_reference("T Pose", "both", width=240, height=310, show_targets=False)
        self.assertEqual(small.shape, (310, 240, 3))
        self.assertIn("LEFT", reference_caption(True))

    def test_sided_poses_differ_between_sides(self):
        l = render_reference("Tree Pose", "left", show_targets=False)
        r = render_reference("Tree Pose", "right", show_targets=False)
        self.assertFalse(np.array_equal(l, r))


class GuidedFlowDefaultsTests(unittest.TestCase):
    def test_default_variant_is_valid_for_every_pose(self):
        """select_pose() in app.py picks 'auto' for sided poses, otherwise the only variant."""
        for name in list_poses():
            spec = get_pose(name)
            default = "auto" if spec.is_sided else spec.variant_names[0]
            opts = ["auto"] + spec.variant_names if spec.is_sided else spec.variant_names
            self.assertIn(default, opts)
            self.assertTrue(resolve_variants(name, default))

    def test_app_initial_state_works_with_pipeline(self):
        """App starts on T Pose / variant 'both'."""
        pipe = YogaPipeline(pose="T Pose", variant="both")
        r = None
        for i in range(30):
            r = pipe.analyze_keypoints(to_pixels(make_pose("T Pose")), i / 15)
        self.assertTrue(r.is_correct)
        self.assertEqual(r.variant, "both")

    def test_auto_detects_the_side_it_explains(self):
        """The README / UI promises Auto finds: Tree = standing leg, Warrior II = front leg, Triangle = hand down."""
        for pose, variant in [("Tree Pose", "left"), ("Tree Pose", "right"), ("Warrior II", "left"),
                              ("Warrior II", "right"), ("Triangle Pose", "left"), ("Triangle Pose", "right")]:
            pipe = YogaPipeline(pose=pose, variant="auto")
            r = None
            for i in range(30):
                r = pipe.analyze_keypoints(to_pixels(make_pose(pose, variant)), i / 15)
            self.assertEqual(r.variant, variant, f"{pose} {variant}")
            self.assertTrue(r.is_correct, f"{pose} {variant}")


if __name__ == "__main__":
    unittest.main()
