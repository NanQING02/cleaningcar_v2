import math
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _install_fake_rknn():
    if "rknnlite.api" in sys.modules:
        return
    pkg = types.ModuleType("rknnlite")
    api = types.ModuleType("rknnlite.api")

    class _RKNNLite:
        pass

    api.RKNNLite = _RKNNLite
    pkg.api = api
    sys.modules["rknnlite"] = pkg
    sys.modules["rknnlite.api"] = api


_install_fake_rknn()

from cleaningcar.plate_lpr import DualPlateRecognizer  # noqa: E402


class _FakeRecognizer:
    @staticmethod
    def inference(inputs, data_format):
        plate_logits = np.array(
            [
                [
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        color_logits = np.array([[0.0, 2.0, 1.0, -1.0, -2.0]], dtype=np.float32)
        return [plate_logits, color_logits]


class PlateLprColorConfTests(unittest.TestCase):
    def test_recognize_returns_color_confidence_by_softmax(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer.recognizer = _FakeRecognizer()

        text, plate_color, color_conf = recognizer._recognize(np.ones((48, 168, 3), dtype=np.uint8))

        self.assertTrue(isinstance(text, str))
        self.assertTrue(plate_color)
        expected = math.exp(2.0) / (
            math.exp(0.0) + math.exp(2.0) + math.exp(1.0) + math.exp(-1.0) + math.exp(-2.0)
        )
        self.assertAlmostEqual(color_conf, expected, places=6)

    def test_infer_frame_includes_plate_color_conf(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer._detect = lambda frame, conf_thresh, iou_thresh: np.array(
            [[0.0, 0.0, 10.0, 10.0, 0.95, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0]],
            dtype=np.float32,
        )
        recognizer._recognize = lambda roi: ("ABC1234", "blue", 0.77)

        with patch("cleaningcar.plate_lpr._four_point_transform", return_value=np.ones((16, 32, 3), dtype=np.uint8)):
            results = recognizer.infer_frame(np.zeros((24, 24, 3), dtype=np.uint8))

        self.assertEqual(len(results), 1)
        self.assertIn("plate_color_conf", results[0])
        self.assertAlmostEqual(float(results[0]["plate_color_conf"]), 0.77, places=6)

    def test_infer_frame_skips_tiny_plate_roi(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer._detect = lambda frame, conf_thresh, iou_thresh: np.array(
            [[0.0, 0.0, 10.0, 10.0, 0.95, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0]],
            dtype=np.float32,
        )
        recognizer._recognize = lambda roi: (_ for _ in ()).throw(AssertionError("tiny roi should be skipped"))

        with patch("cleaningcar.plate_lpr._four_point_transform", return_value=np.ones((3, 6, 3), dtype=np.uint8)):
            results = recognizer.infer_frame(np.zeros((24, 24, 3), dtype=np.uint8))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "")
        self.assertEqual(results[0]["plate_color"], "")
        self.assertEqual(float(results[0]["plate_color_conf"]), 0.0)


if __name__ == "__main__":
    unittest.main()
