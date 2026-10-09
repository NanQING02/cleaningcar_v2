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

    def test_heavy_vehicle_subclasses_are_handoff_compatible(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=30)
        lifecycle = manager.create(1, 'dump truck', capture_ts=100)
        manager.touch(
            1,
            100,
            plate_text='苏C12345',
            plate_box=[10, 10, 30, 30],
            plate_edge='flow_start',
            vehicle_box=[0, 0, 100, 100],
            anchor_point=[50, 100],
        )
        manager.mark_lost(1, capture_ts=101)

        self.assertTrue(manager.can_handoff(
            lifecycle,
            'yellow truck',
            105,
            '苏C12345',
            'flow_start',
            [12, 12, 32, 32],
            True,
        ))

    def test_unique_unplated_spatial_candidate_can_handoff(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=30)
        lifecycle = manager.create(1, 'dump truck', capture_ts=100)
        manager.touch(
            1,
            100,
            vehicle_box=[100, 100, 300, 300],
            anchor_point=[200, 300],
            motion_direction='forward',
        )
        manager.mark_lost(1, capture_ts=101)

        candidates = manager.find_unplated_handoff_candidates(
            'yellow truck',
            120,
            [120, 110, 320, 310],
            anchor_point=[220, 310],
            motion_direction='forward',
        )

        self.assertEqual(candidates, [lifecycle])
        self.assertEqual(
            manager.find_unplated_handoff_candidates(
                'yellow truck',
                120,
                [1000, 1000, 1200, 1200],
                anchor_point=[1100, 1200],
                motion_direction='forward',
            ),
            [],
        )
        self.assertEqual(
            manager.find_unplated_handoff_candidates(
                'yellow truck',
                120,
                [120, 110, 320, 310],
                anchor_point=[220, 310],
                motion_direction='reverse',
            ),
            [],
        )

    def test_conflicting_provisional_plate_identity_rejects_unplated_handoff(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=10, plate_identity_min_hits=2)
        lifecycle = manager.create(1, 'dump truck', capture_ts=100)
        manager.touch(
            1,
            100,
            vehicle_box=[100, 100, 300, 300],
            anchor_point=[200, 300],
            plate_candidate='苏C8596S',
            plate_candidate_hits=2,
        )
        manager.mark_lost(1, capture_ts=101)

        self.assertEqual(
            manager.find_unplated_handoff_candidates(
                'yellow truck',
                102,
                [110, 105, 310, 305],
                anchor_point=[210, 305],
                plate_candidate='苏C7755S',
                plate_candidate_hits=2,
            ),
            [],
        )
        self.assertEqual(
            manager.find_unplated_handoff_candidates(
                'yellow truck',
                102,
                [110, 105, 310, 305],
                anchor_point=[210, 305],
                plate_candidate='苏C8596S',
                plate_candidate_hits=1,
            ),
            [],
        )
        self.assertEqual(
            manager.find_unplated_handoff_candidates(
                'yellow truck',
                102,
                [110, 105, 310, 305],
                anchor_point=[210, 305],
                plate_candidate='苏C8596S',
                plate_candidate_hits=2,
            ),
            [lifecycle],
        )

    def test_locked_plate_only_accepts_short_strict_unplated_continuation(self):
        manager = BusinessLifecycleManager(
            'cam',
            grace_seconds=10,
            locked_plate_unplated_grace_seconds=2,
            locked_plate_unplated_center_scale=1.25,
        )
        lifecycle = manager.create(1, 'dump truck', capture_ts=100)
        manager.touch(
            1,
            100,
            plate_text='苏C7755S',
            plate_box=[120, 120, 180, 150],
            vehicle_box=[100, 100, 300, 300],
            anchor_point=[200, 300],
        )
        manager.mark_lost(1, capture_ts=101)

        self.assertFalse(manager.can_handoff_without_plate(
            lifecycle,
            'yellow truck',
            102,
            [110, 105, 310, 305],
            anchor_point=[210, 305],
            plate_candidate='苏C0566S',
            plate_candidate_hits=2,
        ))
        self.assertTrue(manager.can_handoff_without_plate(
            lifecycle,
            'yellow truck',
            102,
            [110, 105, 310, 305],
            anchor_point=[210, 305],
        ))
        self.assertFalse(manager.can_handoff_without_plate(
            lifecycle,
            'yellow truck',
            104,
            [110, 105, 310, 305],
            anchor_point=[210, 305],
        ))

    def test_handoff_budget_blocks_chained_third_tracker(self):
        manager = BusinessLifecycleManager('cam', grace_seconds=10, max_handoffs=1)
        lifecycle = manager.create(1, 'car', capture_ts=100)
        manager.touch(
            1,
            100,
            plate_text='苏A12345',
            plate_box=[10, 10, 30, 20],
            plate_edge='flow_start',
        )
        manager.mark_lost(1, capture_ts=101)
        self.assertIs(manager.handoff(lifecycle, 2, 102), lifecycle)
        manager.mark_lost(2, capture_ts=103)

        self.assertFalse(manager.can_handoff(
            lifecycle,
            'car',
            104,
            '苏A12345',
            'flow_start',
            [11, 11, 31, 21],
            True,
        ))
        self.assertIsNone(manager.handoff(lifecycle, 3, 104))

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
