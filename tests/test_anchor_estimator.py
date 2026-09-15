import unittest

from cleaningcar.anchor import AnchorEstimator


class AnchorEstimatorTests(unittest.TestCase):
    def test_legacy_mode_is_forced_to_neutral_business_anchor(self):
        estimator = AnchorEstimator(
            frame_size=(1920, 1080),
            flow_vector=((0, 0), (100, 0)),
            logic_cfg={
                'anchor_mode': 'legacy',
                'anchor_offset_ratio': 0.1,
                'anchor_legacy_flow_shift_ratio': 0.3,
            },
        )

        result = estimator.estimate(1, [100, 100, 300, 300], 1)

        self.assertEqual(result.legacy_point, (140.0, 280.0))
        self.assertEqual(result.neutral_point, (200.0, 280.0))
        self.assertEqual(result.directional_point, result.neutral_point)
        self.assertEqual(result.mode, 'neutral')
        self.assertEqual(result.selected_point, result.neutral_point)
        self.assertEqual(result.adaptive_point, (200.0, 284.0))
        self.assertEqual(result.motion_direction, 'unknown')

    def test_ground_diagnostic_is_kept_in_trace_but_cannot_be_selected(self):
        estimator = AnchorEstimator(
            frame_size=(1920, 1080),
            flow_vector=((0, 0), (100, 0)),
            logic_cfg={
                'anchor_mode': 'adaptive_ground',
                'anchor_adaptive_vertical_ratio': 0.08,
                'anchor_adaptive_vertical_cap_ratio': 0.02,
            },
        )

        result = estimator.estimate(1, [100, 50, 900, 900], 1)

        self.assertAlmostEqual(result.vertical_offset_px, 21.6, places=5)
        self.assertAlmostEqual(result.adaptive_point[1], 878.4, places=5)
        self.assertEqual(result.mode, 'neutral')
        self.assertEqual(result.selected_point, result.neutral_point)

    def test_adaptive_point_does_not_depend_on_flow_direction(self):
        box = [100, 100, 300, 300]
        forward = AnchorEstimator((1920, 1080), ((0, 0), (100, 0)), {'anchor_mode': 'directional'})
        reverse = AnchorEstimator((1920, 1080), ((100, 0), (0, 0)), {'anchor_mode': 'directional'})

        self.assertEqual(forward.estimate(1, box, 1).adaptive_point, reverse.estimate(1, box, 1).adaptive_point)
        self.assertNotEqual(forward.estimate(1, box, 1).legacy_point, reverse.estimate(1, box, 1).legacy_point)

    def test_truncated_box_reuses_reliable_motion_history(self):
        estimator = AnchorEstimator(
            frame_size=(1920, 1080),
            flow_vector=((0, 0), (100, 0)),
            logic_cfg={
                'anchor_mode': 'directional',
                'anchor_edge_margin_ratio': 0.01,
                'anchor_reuse_max_frames': 12,
            },
        )
        estimator.estimate(1, [100, 100, 300, 300], 1)
        estimator.estimate(1, [110, 100, 310, 300], 2)

        result = estimator.estimate(1, [120, 400, 320, 1080], 3)

        self.assertTrue(result.box_truncated)
        self.assertIn('bottom', result.truncated_edges)
        self.assertTrue(result.reused_previous)
        self.assertAlmostEqual(result.adaptive_point[0], 220.0, places=5)
        self.assertAlmostEqual(result.adaptive_point[1], 284.0, places=5)
        self.assertLess(result.confidence, 1.0)

    def test_same_track_frame_is_cached_once(self):
        estimator = AnchorEstimator((1920, 1080), ((0, 0), (100, 0)), {'anchor_mode': 'legacy'})

        first = estimator.estimate(1, [100, 100, 300, 300], 1)
        second = estimator.estimate(1, [120, 120, 320, 320], 1)

        self.assertIs(first, second)
        self.assertEqual(len(estimator.get_history(1)), 1)

    def test_forward_motion_locks_from_neutral_history_and_moves_d_toward_l(self):
        estimator = AnchorEstimator(
            (1920, 1080),
            ((0, 0), (100, 0)),
            {
                'anchor_mode': 'legacy',
                'anchor_direction_window': 8,
                'anchor_direction_min_points': 5,
                'anchor_direction_consistency': 0.7,
                'anchor_direction_blend_frames': 5,
            },
        )

        results = [
            estimator.estimate(1, [100 + step * 10, 100, 300 + step * 10, 300], step + 1)
            for step in range(9)
        ]

        self.assertEqual(results[3].motion_direction, 'unknown')
        self.assertEqual(results[3].directional_point, results[3].neutral_point)
        self.assertEqual(results[4].motion_direction, 'forward')
        self.assertTrue(results[4].direction_locked)
        self.assertAlmostEqual(results[4].direction_blend, 0.2)
        self.assertGreater(results[4].directional_point[0], results[4].legacy_point[0])
        self.assertEqual(results[-1].directional_point, results[-1].legacy_point)
        self.assertEqual(results[-1].selected_point, results[-1].neutral_point)

    def test_reverse_motion_mirrors_directional_point_across_neutral_point(self):
        estimator = AnchorEstimator(
            (1920, 1080),
            ((0, 0), (100, 0)),
            {
                'anchor_mode': 'legacy',
                'anchor_direction_min_points': 5,
                'anchor_direction_blend_frames': 1,
            },
        )

        result = None
        for step in range(5):
            result = estimator.estimate(
                1,
                [200 - step * 10, 100, 400 - step * 10, 300],
                step + 1,
            )

        self.assertEqual(result.motion_direction, 'reverse')
        self.assertLess(result.legacy_point[0], result.neutral_point[0])
        self.assertGreater(result.directional_point[0], result.neutral_point[0])
        self.assertAlmostEqual(
            result.directional_point[0] - result.neutral_point[0],
            result.neutral_point[0] - result.legacy_point[0],
        )

    def test_inconsistent_motion_keeps_direction_unknown(self):
        estimator = AnchorEstimator(
            (1920, 1080),
            ((0, 0), (100, 0)),
            {
                'anchor_direction_min_points': 5,
                'anchor_direction_consistency': 0.7,
            },
        )

        result = None
        for frame_idx, center_x in enumerate((200, 210, 205, 215, 210), start=1):
            result = estimator.estimate(
                1,
                [center_x - 100, 100, center_x + 100, 300],
                frame_idx,
            )

        self.assertEqual(result.motion_direction, 'unknown')
        self.assertFalse(result.direction_locked)
        self.assertEqual(result.directional_point, result.neutral_point)

    def test_directional_diagnostic_never_moves_business_anchor(self):
        estimator = AnchorEstimator(
            (1920, 1080),
            ((0, 0), (100, 0)),
            {
                'anchor_mode': 'directional',
                'anchor_direction_min_points': 5,
                'anchor_direction_blend_frames': 1,
            },
        )

        result = None
        for step in range(5):
            result = estimator.estimate(1, [100 + step * 10, 100, 300 + step * 10, 300], step + 1)

        self.assertEqual(result.directional_point, result.legacy_point)
        self.assertEqual(result.selected_point, result.neutral_point)
        self.assertNotEqual(result.selected_point, result.directional_point)

    def test_ground_history_expiry_does_not_clear_direction_lock(self):
        estimator = AnchorEstimator(
            (1920, 1080),
            ((0, 0), (100, 0)),
            {
                'anchor_direction_min_points': 5,
                'anchor_direction_blend_frames': 1,
                'anchor_edge_margin_ratio': 0.01,
            },
        )
        for frame_idx in range(1, 6):
            estimator.estimate(
                1,
                [100 + frame_idx * 10, 100, 300 + frame_idx * 10, 300],
                frame_idx,
            )
        for frame_idx in range(6, 9):
            estimator.estimate(
                1,
                [100 + frame_idx * 10, 400, 300 + frame_idx * 10, 1080],
                frame_idx,
            )

        estimator.cleanup(frame_idx=8, max_age=2)
        result = estimator.estimate(1, [190, 400, 390, 1080], 9)

        self.assertEqual(result.motion_direction, 'forward')
        self.assertTrue(result.direction_locked)


if __name__ == '__main__':
    unittest.main()
