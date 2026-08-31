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

from cleaningcar.constants import PLATE_DECODE_CHARS  # noqa: E402
from cleaningcar.plate_lpr import DualPlateRecognizer  # noqa: E402


class _FakeRecognizer:
    @staticmethod
    def inference(inputs, data_format):
        plate_logits = np.zeros((1, 2, len(PLATE_DECODE_CHARS)), dtype=np.float32)
        plate_logits[0, 0, 1] = 2.0
        plate_logits[0, 1, 2] = 3.0
        color_logits = np.array([[0.0, 2.0, 1.0, -1.0, -2.0]], dtype=np.float32)
        return [plate_logits, color_logits]


class _BadShapeDetector:
    def __init__(self):
        self.calls = 0

    def inference(self, inputs, data_format):
        self.calls += 1
        return [np.zeros((1, 14), dtype=np.float32)]


class PlateLprColorConfTests(unittest.TestCase):
    def test_recognize_returns_color_confidence_by_softmax(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer.recognizer = _FakeRecognizer()

        text, plate_color, color_conf, text_conf = recognizer._recognize(np.ones((48, 168, 3), dtype=np.uint8))

        self.assertTrue(isinstance(text, str))
        self.assertTrue(plate_color)
        expected = math.exp(2.0) / (
            math.exp(0.0) + math.exp(2.0) + math.exp(1.0) + math.exp(-1.0) + math.exp(-2.0)
        )
        self.assertAlmostEqual(color_conf, expected, places=6)
        self.assertGreater(text_conf, 0.0)

    def test_infer_frame_includes_plate_color_conf(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer._detect = lambda frame, conf_thresh, iou_thresh: np.array(
            [[0.0, 0.0, 10.0, 10.0, 0.95, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0]],
            dtype=np.float32,
        )
        recognizer._recognize = lambda roi: ("ABC1234", "blue", 0.77, 0.91)

        with patch("cleaningcar.plate_lpr._four_point_transform", return_value=np.ones((16, 32, 3), dtype=np.uint8)):
            results = recognizer.infer_frame(np.zeros((24, 24, 3), dtype=np.uint8))

        self.assertEqual(len(results), 1)
        self.assertIn("plate_color_conf", results[0])
        self.assertAlmostEqual(float(results[0]["plate_color_conf"]), 0.77, places=6)
        self.assertAlmostEqual(float(results[0]["plate_text_conf"]), 0.91, places=6)

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

    def test_detect_logs_shape_once_and_skips_bad_shape(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer.detector = _BadShapeDetector()
        recognizer.log_output_shape_once = True
        recognizer._detect_shape_logged = False
        recognizer._rec_shape_logged = False
        recognizer._detect_shape_warned = False
        recognizer._rec_shape_warned = False

        with patch("builtins.print") as mocked_print:
            first = recognizer._detect(np.zeros((24, 24, 3), dtype=np.uint8), 0.3, 0.5)
            second = recognizer._detect(np.zeros((24, 24, 3), dtype=np.uint8), 0.3, 0.5)

        self.assertEqual(first.shape, (0, 14))
        self.assertEqual(second.shape, (0, 14))
        printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertEqual(printed.count("detect output shapes"), 1)
        self.assertEqual(printed.count("detect output shape warning"), 1)


if __name__ == "__main__":
    unittest.main()
