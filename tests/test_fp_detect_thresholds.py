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
