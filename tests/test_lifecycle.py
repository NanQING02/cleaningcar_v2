import unittest

from cleaningcar.lifecycle import BusinessLifecycleManager


class BusinessLifecycleManagerTests(unittest.TestCase):
    def test_event_id_fits_platform_database_limit_for_long_camera_name(self):
        manager = BusinessLifecycleManager('RK3588-DEV-绕行', grace_seconds=8)

        lifecycle = manager.create(1, 'car', capture_ts=1789482227.779)

        self.assertLessEqual(len(lifecycle.event_id), 36)
        self.assertTrue(lifecycle.event_id.startswith('RK3588-DEV-'))

    def test_active_lifecycle_cannot_be_handed_off(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        lifecycle.last_plate_ts = 100
        lifecycle.last_plate_text = '鲁A12345'
        lifecycle.last_plate_edge = 'bottom'

        self.assertFalse(manager.can_handoff(
            lifecycle, 'car', 101, '鲁A12345', 'bottom', [0, 0, 20, 20], True,
        ))

    def test_lost_lifecycle_handoffs_only_inside_plate_window(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        lifecycle.last_plate_ts = 101
        lifecycle.last_plate_box = (0, 0, 20, 20)
        lifecycle.last_plate_text = '鲁A12345'
        lifecycle.last_plate_edge = 'bottom'
        manager.mark_lost(1, capture_ts=102)

        self.assertTrue(manager.can_handoff(
            lifecycle, 'car', 109, '鲁A12345', 'bottom', [1, 1, 21, 21], True,
        ))
        manager.handoff(lifecycle, 2, 109)
        self.assertIs(manager.get(2), lifecycle)
        self.assertEqual(lifecycle.tracker_ids, {1, 2})

    def test_finalized_or_wrong_edge_lifecycle_cannot_handoff(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        lifecycle.last_plate_ts = 100
        lifecycle.last_plate_text = '鲁A12345'
        lifecycle.last_plate_edge = 'bottom'
        manager.mark_lost(1, capture_ts=101)

        self.assertFalse(manager.can_handoff(
            lifecycle, 'car', 102, '鲁A12345', 'top', [0, 0, 20, 20], True,
        ))
        manager.finalize(1)
        self.assertFalse(manager.can_handoff(
            lifecycle, 'car', 102, '鲁A12345', 'bottom', [0, 0, 20, 20], True,
        ))

    def test_handoff_requires_unique_spatially_continuous_candidate(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        first = manager.create(1, 'car', capture_ts=100)
        manager.touch(1, 100, plate_text='鲁A12345', plate_box=[0, 0, 20, 20], plate_edge='flow_start')
        manager.mark_lost(1, capture_ts=101)
        second = manager.create(2, 'car', capture_ts=100)
        manager.touch(2, 100, plate_text='鲁A54321', plate_box=[2, 2, 22, 22], plate_edge='flow_start')
        manager.mark_lost(2, capture_ts=101)

        candidates = manager.find_handoff_candidates(
            'car', 102, '鲁A12345', 'flow_start', [3, 3, 23, 23], True,
        )

        self.assertEqual(candidates, [first])
        self.assertEqual(
            manager.find_handoff_candidates(
                'car', 102, '鲁A12345', 'flow_start', [500, 500, 520, 520], True,
            ),
            [],
        )

    def test_different_stable_plate_cannot_handoff(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        manager.touch(
            1,
            100,
            plate_text='鲁A12345',
            plate_box=[0, 0, 20, 20],
            plate_edge='flow_start',
        )
        manager.mark_lost(1, capture_ts=101)

        self.assertFalse(manager.can_handoff(
            lifecycle,
            'car',
            102,
            '鲁A54321',
            'flow_start',
            [1, 1, 21, 21],
            True,
        ))

    def test_waiting_lifecycles_only_returns_same_class_inside_grace(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        car = manager.create(1, 'car', capture_ts=100)
        manager.touch(
            1,
            100,
            plate_text='鲁A12345',
            plate_box=[0, 0, 20, 20],
            plate_edge='flow_start',
        )
        manager.mark_lost(1, capture_ts=101)
        truck = manager.create(2, 'yellow truck', capture_ts=100)
        manager.touch(
            2,
            100,
            plate_text='鲁A54321',
            plate_box=[0, 0, 20, 20],
            plate_edge='flow_start',
        )
        manager.mark_lost(2, capture_ts=101)

        self.assertEqual(manager.find_waiting_lifecycles('car', 108), [car])
        self.assertEqual(manager.find_waiting_lifecycles('car', 110), [])
        self.assertNotIn(truck, manager.find_waiting_lifecycles('car', 108))

    def test_closed_lifecycles_are_pruned_after_retention_window(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        manager.handoff(manager.mark_lost(1, capture_ts=101), 2, 102)
        manager.finalize(2, capture_ts=110)

        self.assertEqual(manager.cleanup(capture_ts=169, closed_retention_seconds=60), 0)
        self.assertIs(manager.get(1), lifecycle)
        self.assertEqual(manager.cleanup(capture_ts=170, closed_retention_seconds=60), 1)
        self.assertIsNone(manager.get(1))
        self.assertIsNone(manager.get(2))
        self.assertEqual(manager.snapshot()['total'], 0)

    def test_cleanup_keeps_active_lifecycle(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=8)
        manager.create(1, 'car', capture_ts=100)

        self.assertEqual(manager.cleanup(capture_ts=10000, closed_retention_seconds=60), 0)
        self.assertEqual(manager.snapshot()['active'], 1)
