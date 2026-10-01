import unittest

import numpy as np

from cleaningcar.fp_detect import FpModelPostprocessor


class FpDetectThresholdTests(unittest.TestCase):
    def test_class_thresholds_filter_candidates_before_nms(self):
        postprocessor = FpModelPostprocessor(
            obj_thresh=0.25,
            num_classes=3,
            class_thresholds={0: 0.8, 1: 0.4},
        )
        scores = np.array([0.5, 0.85, 0.3, 0.45], dtype=np.float32)
        classes = np.array([0, 0, 1, 2], dtype=np.int32)

        np.testing.assert_array_equal(
            postprocessor.candidate_keep_mask(scores, classes),
            np.array([False, True, False, True]),
        )

    def test_default_threshold_behavior_is_unchanged(self):
        postprocessor = FpModelPostprocessor(obj_thresh=0.25, num_classes=2)
        scores = np.array([0.24, 0.25], dtype=np.float32)
        classes = np.array([0, 1], dtype=np.int32)

        np.testing.assert_array_equal(
            postprocessor.candidate_keep_mask(scores, classes),
            np.array([False, True]),
        )

    def test_vehicle_classes_share_class_agnostic_nms_group(self):
        postprocessor = FpModelPostprocessor(
            obj_thresh=0.25,
            nms_thresh=0.45,
            num_classes=3,
            class_agnostic_groups=[{0, 1}],
        )
        boxes = np.array([
            [0.0, 0.0, 100.0, 100.0],
            [2.0, 2.0, 102.0, 102.0],
            [200.0, 200.0, 250.0, 250.0],
        ], dtype=np.float32)
        classes = np.array([0, 1, 2], dtype=np.int32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        kept_boxes, kept_classes, kept_scores = postprocessor.apply_nms(
            boxes, classes, scores,
        )

        self.assertEqual(kept_boxes.shape[0], 2)
        self.assertEqual(set(kept_classes.tolist()), {0, 2})
        np.testing.assert_allclose(sorted(kept_scores.tolist()), [0.7, 0.9])
