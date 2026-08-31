import unittest

from zone_manager import ZoneManager


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

    def test_initial_core_uses_compatibility_entry_after_three_hits(self):
        manager = self._manager()

        for frame_idx in (1, 2):
            state, flags = manager.update_track(1, (50, 50), frame_idx, vehicle_height=40)
            self.assertFalse(flags['enter_a'])
        state, flags = manager.update_track(1, (50, 50), 3, vehicle_height=40)

        self.assertTrue(flags['enter_a'])
        self.assertTrue(state.inside_a)
        self.assertTrue(state.initial_core_compat)
        self.assertEqual(state.transition_reason, 'enter_a_initial_core_compat')


if __name__ == '__main__':
    unittest.main()
