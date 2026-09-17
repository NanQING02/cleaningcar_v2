import unittest

from zone_manager import ZoneManager, ZoneState


class ZoneManagerDebounceTests(unittest.TestCase):
    @staticmethod
    def _manager():
        return ZoneManager(
            zone_a=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_b=[],
            flow_vector=((0, 50), (100, 50)),
            zone_a_margin_ratio=0.10,
            zone_a_margin_min_px=4,
            zone_a_margin_max_px=24,
            zone_a_observed_outside_hits=3,
            zone_a_enter_core_hits=3,
            zone_a_exit_outside_hits=5,
        )

    def test_signed_distance_and_dynamic_margin_are_clamped(self):
        manager = self._manager()

        outside, _ = manager.update_track(1, (-30, 50), 1, vehicle_height=20)
        core, _ = manager.update_track(2, (50, 50), 1, vehicle_height=400)

        self.assertEqual(outside.zone_a_region, 'OUTSIDE')
        self.assertAlmostEqual(outside.signed_distance, -30.0)
        self.assertEqual(outside.dynamic_margin, 4.0)
        self.assertEqual(core.zone_a_region, 'CORE')
        self.assertAlmostEqual(core.signed_distance, 50.0)
        self.assertEqual(core.dynamic_margin, 24.0)

    def test_counts_unique_valid_observations_instead_of_frame_gaps(self):
        manager = self._manager()

        state, _ = manager.update_track(1, (-10, 50), 1, vehicle_height=40)
        duplicate, duplicate_flags = manager.update_track(1, (-10, 50), 1, vehicle_height=40)
        state, _ = manager.update_track(1, (-10, 50), 100, vehicle_height=40)
        state, _ = manager.update_track(1, (-10, 50), 500, vehicle_height=40)

        self.assertIs(state, duplicate)
        self.assertEqual(duplicate_flags, {
            'enter_a': False,
            'exit_a': False,
            'enter_b': False,
            'exit_b': False,
        })
        self.assertEqual(state.observed_outside_count, 3)
        self.assertEqual(state.zone_a_state, 'OBSERVED_OUTSIDE')

    def test_invalid_vehicle_height_does_not_advance_zone_a(self):
        manager = self._manager()

        state, flags = manager.update_track(1, (-10, 50), 1, vehicle_height=None)
        state, valid_flags = manager.update_track(1, (-10, 50), 1, vehicle_height=40)

        self.assertEqual(state.zone_a_region, 'OUTSIDE')
        self.assertEqual(state.observed_outside_count, 1)
        self.assertFalse(flags['enter_a'])
        self.assertFalse(flags['exit_a'])
        self.assertFalse(valid_flags['enter_a'])
        self.assertFalse(valid_flags['exit_a'])

    def test_buffer_preserves_last_clear_candidate_and_resets_opposite(self):
        manager = self._manager()

        for frame_idx in (1, 2, 3):
            state, _ = manager.update_track(1, (-10, 50), frame_idx, vehicle_height=40)
        self.assertEqual(state.zone_a_state, 'OBSERVED_OUTSIDE')

        state, _ = manager.update_track(1, (2, 50), 4, vehicle_height=40)
        self.assertEqual(state.observed_outside_count, 3)
        self.assertEqual(state.enter_core_count, 0)

        state, _ = manager.update_track(1, (10, 50), 5, vehicle_height=40)
        state, _ = manager.update_track(1, (2, 50), 6, vehicle_height=40)
        self.assertEqual(state.enter_core_count, 1)
        self.assertEqual(state.observed_outside_count, 0)

        _, flags = manager.update_track(1, (10, 50), 7, vehicle_height=40)
        state, flags = manager.update_track(1, (10, 50), 8, vehicle_height=40)
        self.assertTrue(flags['enter_a'])
        self.assertTrue(state.inside_a)

        state, _ = manager.update_track(1, (-10, 50), 9, vehicle_height=40)
        state, _ = manager.update_track(1, (2, 50), 10, vehicle_height=40)
        self.assertEqual(state.exit_outside_count, 1)
        for frame_idx in (11, 12, 13):
            state, flags = manager.update_track(1, (-10, 50), frame_idx, vehicle_height=40)
            self.assertFalse(flags['exit_a'])
        state, flags = manager.update_track(1, (-10, 50), 14, vehicle_height=40)
        self.assertTrue(flags['exit_a'])
        self.assertFalse(state.inside_a)

        for frame_idx in (15, 16):
            state, flags = manager.update_track(1, (10, 50), frame_idx, vehicle_height=40)
            self.assertFalse(flags['enter_a'])
        state, flags = manager.update_track(1, (10, 50), 17, vehicle_height=40)
        self.assertTrue(flags['enter_a'])
        self.assertTrue(state.inside_a)
        self.assertIsNone(state.exit_ratio)
        self.assertEqual(state.exit_point, (0.0, 0.0))

    def test_auto_reference_edge_uses_zone_parallel_midline(self):
        manager = ZoneManager(
            zone_a=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_b=[],
            flow_vector=((50, 0), (50, 100)),
            direction_reference_edge='auto',
        )

        self.assertEqual(manager.direction_reference_edge_index, 0)
        self.assertAlmostEqual(manager._relative_position((50, 25)), 0.25)
        self.assertAlmostEqual(manager._relative_position((50, 75)), 0.75)

    def test_manual_reference_edge_controls_direction_axis(self):
        manager = ZoneManager(
            zone_a=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_b=[],
            flow_vector=((0, 50), (100, 50)),
            direction_reference_edge=1,
        )

        self.assertEqual(manager.direction_reference_edge_index, 1)
        self.assertAlmostEqual(manager._relative_position((25, 50)), 0.25)
        self.assertAlmostEqual(manager._relative_position((75, 50)), 0.75)

    def test_direction_keeps_four_entry_exit_combinations(self):
        manager = self._manager()
        cases = (
            (0.25, 0.75, (5, '正向前出')),
            (0.25, 0.25, (6, '正向反出')),
            (0.75, 0.75, (7, '反向前出')),
            (0.75, 0.25, (8, '反向反出')),
        )

        for entry_ratio, exit_ratio, expected in cases:
            self.assertEqual(
                manager.resolve_direction(ZoneState(enter_ratio=entry_ratio, exit_ratio=exit_ratio)),
                expected,
            )

    def test_initial_core_becomes_born_inside_candidate_after_three_hits(self):
        manager = self._manager()

        for frame_idx in (1, 2):
            state, flags = manager.update_track(1, (50, 50), frame_idx, vehicle_height=40)
            self.assertFalse(flags['enter_a'])
        state, flags = manager.update_track(1, (50, 50), 3, vehicle_height=40)

        self.assertTrue(flags['enter_a'])
        self.assertTrue(state.inside_a)
        self.assertTrue(state.initial_core_compat)
        self.assertTrue(state.born_inside_a)
        self.assertTrue(state.born_inside_pending)
        self.assertEqual(state.born_inside_outcome, 'CANDIDATE')
        self.assertEqual(state.transition_reason, 'born_inside_a_candidate')

    def test_born_inside_promotes_after_zone_b_entry(self):
        manager = ZoneManager(
            zone_a=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_b=[(40, 0), (100, 0), (100, 100), (40, 100)],
            flow_vector=((0, 50), (100, 50)),
            entry_hysteresis=3,
            zone_a_enter_core_hits=3,
        )

        for frame_idx in (1, 2):
            state, _ = manager.update_track(1, (50, 50), frame_idx, vehicle_height=40)
        state, flags = manager.update_track(1, (50, 50), 3, vehicle_height=40)

        self.assertTrue(flags['enter_b'])
        self.assertTrue(state.born_inside_a)
        self.assertFalse(state.born_inside_pending)
        self.assertEqual(state.born_inside_outcome, 'PROMOTED')
        self.assertEqual(state.transition_reason, 'born_inside_promoted_by_zone_b')

    def test_zone_b_exit_ignores_anchor_jitter_inside_dynamic_margin(self):
        manager = ZoneManager(
            zone_a=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_b=[(0, 0), (100, 0), (100, 100), (0, 100)],
            flow_vector=((0, 50), (100, 50)),
            entry_hysteresis=1,
            exit_hysteresis=3,
            zone_a_margin_ratio=0.10,
            zone_a_margin_min_px=4,
            zone_a_margin_max_px=24,
        )

        state, flags = manager.update_track(1, (50, 50), 1, vehicle_height=40)
        self.assertTrue(flags['enter_b'])
        self.assertTrue(state.inside_b)

        for frame_idx in (2, 3, 4, 5):
            state, flags = manager.update_track(1, (-2, 50), frame_idx, vehicle_height=40)
            self.assertEqual(state.zone_b_region, 'BUFFER')
            self.assertFalse(flags['exit_b'])
            self.assertTrue(state.inside_b)

        for frame_idx in (6, 7):
            state, flags = manager.update_track(1, (-10, 50), frame_idx, vehicle_height=40)
            self.assertFalse(flags['exit_b'])
        state, flags = manager.update_track(1, (-10, 50), 8, vehicle_height=40)

        self.assertTrue(flags['exit_b'])
        self.assertFalse(state.inside_b)

    def test_born_inside_exits_without_zone_b_as_pass_by(self):
        manager = self._manager()

        for frame_idx in (1, 2, 3):
            state, _ = manager.update_track(1, (50, 50), frame_idx, vehicle_height=40)
        for frame_idx in (4, 5, 6, 7):
            state, flags = manager.update_track(1, (-10, 50), frame_idx, vehicle_height=40)
            self.assertFalse(flags['exit_a'])
        state, flags = manager.update_track(1, (-10, 50), 8, vehicle_height=40)

        self.assertTrue(flags['exit_a'])
        self.assertFalse(state.born_inside_pending)
        self.assertEqual(state.born_inside_outcome, 'PASS_BY')
        self.assertEqual(state.transition_reason, 'born_inside_pass_by')


if __name__ == '__main__':
    unittest.main()
