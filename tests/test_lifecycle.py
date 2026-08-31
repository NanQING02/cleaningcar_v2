import unittest

from cleaningcar.lifecycle import BusinessLifecycleManager


class BusinessLifecycleManagerTests(unittest.TestCase):
    def test_active_lifecycle_cannot_be_handed_off(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        lifecycle.last_plate_ts = 100
        lifecycle.last_plate_edge = 'bottom'

        self.assertFalse(manager.can_handoff(lifecycle, 'car', 101, 'bottom', [0, 0, 20, 20], True))

    def test_lost_lifecycle_handoffs_only_inside_plate_window(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        lifecycle.last_plate_ts = 101
        lifecycle.last_plate_box = (0, 0, 20, 20)
        lifecycle.last_plate_edge = 'bottom'
        manager.mark_lost(1, capture_ts=102)

        self.assertTrue(manager.can_handoff(lifecycle, 'car', 109, 'bottom', [1, 1, 21, 21], True))
        manager.handoff(lifecycle, 2, 109)
        self.assertIs(manager.get(2), lifecycle)
        self.assertEqual(lifecycle.tracker_ids, {1, 2})

    def test_finalized_or_wrong_edge_lifecycle_cannot_handoff(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        lifecycle.last_plate_ts = 100
        lifecycle.last_plate_edge = 'bottom'
        manager.mark_lost(1, capture_ts=101)

        self.assertFalse(manager.can_handoff(lifecycle, 'car', 102, 'top', [0, 0, 20, 20], True))
        manager.finalize(1)
        self.assertFalse(manager.can_handoff(lifecycle, 'car', 102, 'bottom', [0, 0, 20, 20], True))

    def test_handoff_requires_unique_spatially_continuous_candidate(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        first = manager.create(1, 'car', capture_ts=100)
        manager.touch(1, 100, plate_text='鲁A12345', plate_box=[0, 0, 20, 20], plate_edge='flow_start')
        manager.mark_lost(1, capture_ts=101)
        second = manager.create(2, 'car', capture_ts=100)
        manager.touch(2, 100, plate_text='鲁A54321', plate_box=[2, 2, 22, 22], plate_edge='flow_start')
        manager.mark_lost(2, capture_ts=101)

        candidates = manager.find_handoff_candidates(
            'car', 102, 'flow_start', [3, 3, 23, 23], True,
        )

        self.assertEqual(candidates, [first, second])
        self.assertEqual(
            manager.find_handoff_candidates('car', 102, 'flow_start', [500, 500, 520, 520], True),
            [],
        )
