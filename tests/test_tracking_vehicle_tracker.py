import unittest

from cleaningcar.tracking import VehicleTracker


class VehicleTrackerTests(unittest.TestCase):
    def test_default_impl_is_bytetrack(self):
        tracker = VehicleTracker(iou_thresh=0.3, max_age=30, center_gate_ratio=0.0)
        self.assertEqual(getattr(tracker, "impl", ""), "bytetrack")

    def test_bytetrack_keeps_id_after_short_occlusion(self):
        tracker = VehicleTracker(iou_thresh=0.3, max_age=30, center_gate_ratio=0.0)

        dets_f1 = [{"box": [0, 0, 10, 10], "score": 0.95, "cls": 0}]
        ids_f1 = tracker.update(1, dets_f1)
        self.assertEqual(len(ids_f1), 1)
        first_id = ids_f1[0]
        self.assertGreater(first_id, 0)

        dets_f2 = [{"box": [5, 0, 15, 10], "score": 0.95, "cls": 0}]
        ids_f2 = tracker.update(2, dets_f2)
        self.assertEqual(ids_f2[0], first_id)

        tracker.update(3, [])

        dets_f4 = [{"box": [12, 0, 22, 10], "score": 0.95, "cls": 0}]
        ids_f4 = tracker.update(4, dets_f4)
        self.assertEqual(ids_f4[0], first_id)

    def test_legacy_impl_can_switch_back(self):
        tracker = VehicleTracker(
            iou_thresh=0.3,
            max_age=30,
            center_gate_ratio=0.0,
            tracker_impl="legacy",
        )
        self.assertEqual(getattr(tracker, "impl", ""), "legacy")

        dets_f1 = [{"box": [0, 0, 10, 10], "score": 0.95, "cls": 0}]
        ids_f1 = tracker.update(1, dets_f1)
        first_id = ids_f1[0]
        self.assertGreater(first_id, 0)

        tracker.update(2, [])

        dets_f3 = [{"box": [30, 0, 40, 10], "score": 0.95, "cls": 0}]
        ids_f3 = tracker.update(3, dets_f3)
        self.assertNotEqual(ids_f3[0], first_id)
        self.assertGreater(ids_f3[0], first_id)


if __name__ == "__main__":
    unittest.main()
