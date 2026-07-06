import unittest

from cleaningcar.video_io import (
    core_mask_to_indices,
    indices_to_core_mask,
    resolve_auto_plate_core_mask,
    resolve_worker_core_masks,
)


class VideoIoCoreMaskAssignmentTests(unittest.TestCase):
    def test_core_mask_round_trip_helpers(self):
        self.assertEqual(core_mask_to_indices(0b111), [0, 1, 2])
        self.assertEqual(indices_to_core_mask([0, 2]), 0b101)

    def test_resolve_worker_core_masks_auto_splits_multi_core_mask(self):
        self.assertEqual(resolve_worker_core_masks(0b111, 2, strategy='auto'), [0b001, 0b010])

    def test_resolve_worker_core_masks_share_mode_preserves_shared_mask(self):
        self.assertEqual(resolve_worker_core_masks(0b011, 2, strategy='share'), [0b011, 0b011])

    def test_resolve_auto_plate_core_mask_uses_spare_core(self):
        worker_masks = resolve_worker_core_masks(0b111, 2, strategy='auto')
        self.assertEqual(resolve_auto_plate_core_mask(None, 0b111, worker_masks), 0b100)

    def test_resolve_auto_plate_core_mask_respects_explicit_request(self):
        worker_masks = resolve_worker_core_masks(0b111, 2, strategy='auto')
        self.assertEqual(resolve_auto_plate_core_mask(0b010, 0b111, worker_masks), 0b010)

    def test_resolve_auto_plate_core_mask_avoids_reserved_masks(self):
        worker_masks = resolve_worker_core_masks(0b111, 2, strategy='auto')
        self.assertIsNone(resolve_auto_plate_core_mask(None, 0b111, worker_masks, reserved_masks=[0b100]))


if __name__ == "__main__":
    unittest.main()
