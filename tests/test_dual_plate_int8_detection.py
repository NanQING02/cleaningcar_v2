import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _install_fake_rknn():
    if 'rknnlite.api' in sys.modules:
        return
    package = types.ModuleType('rknnlite')
    api = types.ModuleType('rknnlite.api')

    class _RKNNLite:
        pass

    api.RKNNLite = _RKNNLite
    package.api = api
    sys.modules['rknnlite'] = package
    sys.modules['rknnlite.api'] = api


_install_fake_rknn()

from cleaningcar.plate_lpr import DualPlateRecognizer, _decode_int8_detection_heads  # noqa: E402


class _RawHeadDetector:
    def __init__(self, outputs):
        self.outputs = outputs

    def inference(self, inputs, data_format):
        del inputs, data_format
        return self.outputs


def _raw_head_outputs():
    outputs = [
        np.full((1, 45, 80, 80), -20.0, dtype=np.float32),
        np.full((1, 45, 40, 40), -20.0, dtype=np.float32),
        np.full((1, 45, 20, 20), -20.0, dtype=np.float32),
    ]
    row, column = 20, 30
    outputs[0][0, 4, row, column] = 10.0
    outputs[0][0, 13, row, column] = 10.0
    return outputs


class DualPlateInt8DetectionTests(unittest.TestCase):
    def test_raw_heads_decode_to_project_detection_contract(self):
        detections = _decode_int8_detection_heads(
            _raw_head_outputs(),
            conf_thresh=0.3,
            iou_thresh=0.5,
            scale=1.0,
            left=0,
            top=0,
        )

        self.assertEqual(detections.shape, (1, 14))
        self.assertGreater(float(detections[0, 4]), 0.9)
        self.assertEqual(int(detections[0, -1]), 0)

    def test_recognizer_dispatches_three_int8_detection_heads(self):
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer.detector = _RawHeadDetector(_raw_head_outputs())
        recognizer.log_output_shape_once = False
        recognizer._detect_shape_logged = False
        recognizer._detect_shape_warned = False

        detections = recognizer._detect(np.zeros((640, 640, 3), dtype=np.uint8), 0.3, 0.5)

        self.assertEqual(detections.shape, (1, 14))
        self.assertGreater(float(detections[0, 4]), 0.9)

    def test_recognizer_warns_and_rejects_invalid_raw_head_schema(self):
        invalid_outputs = [
            np.zeros((1, 44, 80, 80), dtype=np.float32),
            np.zeros((1, 44, 40, 40), dtype=np.float32),
            np.zeros((1, 44, 20, 20), dtype=np.float32),
        ]
        recognizer = DualPlateRecognizer.__new__(DualPlateRecognizer)
        recognizer.detector = _RawHeadDetector(invalid_outputs)
        recognizer.log_output_shape_once = True
        recognizer._detect_shape_logged = False
        recognizer._detect_shape_warned = False

        with patch('builtins.print') as mocked_print:
            detections = recognizer._detect(np.zeros((640, 640, 3), dtype=np.uint8), 0.3, 0.5)

        self.assertEqual(detections.shape, (0, 14))
        printed = '\n'.join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertIn('unsupported INT8 raw-head schema', printed)


if __name__ == '__main__':
    unittest.main()
