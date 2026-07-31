import queue
import sys
import types
import unittest
from types import SimpleNamespace

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

from cleaningcar.constants import LICENSE_CLASS  # noqa: E402
from cleaningcar.worker import DetectWorker  # noqa: E402


class _FakeRK:
    @staticmethod
    def inference(inputs, data_format):
        return [np.array([1.0], dtype=np.float32)]


class _FailingRK:
    @staticmethod
    def inference(inputs, data_format):
        raise RuntimeError("boom")


class _FakePostprocessor:
    @staticmethod
    def prepare(frame):
        return np.zeros((1, 16, 16, 3), dtype=np.uint8), {"dummy": 1}

    @staticmethod
    def postprocess(outputs):
        boxes = np.array([[0.0, 0.0, 20.0, 20.0]], dtype=np.float32)
        classes = np.array([0], dtype=np.int32)
        scores = np.array([0.95], dtype=np.float32)
        return boxes, classes, scores

    @staticmethod
    def map_boxes_to_original(boxes, lb_info):
        return boxes


class _FakeWaterOnlyPostprocessor:
    @staticmethod
    def prepare(frame):
        return np.zeros((1, 16, 16, 3), dtype=np.uint8), {"dummy": 1}

    @staticmethod
    def postprocess(outputs):
        boxes = np.array([[0.0, 0.0, 20.0, 20.0]], dtype=np.float32)
        classes = np.array([6], dtype=np.int32)
        scores = np.array([0.95], dtype=np.float32)
        return boxes, classes, scores

    @staticmethod
    def map_boxes_to_original(boxes, lb_info):
        return boxes


class _FakeNoDetectionsPostprocessor:
    @staticmethod
    def prepare(frame):
        return np.zeros((1, 16, 16, 3), dtype=np.uint8), {"dummy": 1}

    @staticmethod
    def postprocess(outputs):
        boxes = np.array([[0.0, 0.0, 20.0, 20.0]], dtype=np.float32)
        classes = np.array([0], dtype=np.int32)
        scores = np.array([0.01], dtype=np.float32)
        return boxes, classes, scores

    @staticmethod
    def map_boxes_to_original(boxes, lb_info):
        return boxes


class _FakeDualLpr:
    @staticmethod
    def infer_frame(frame, conf_thresh, iou_thresh):
        return [
            {
                "box": [1.0, 2.0, 11.0, 12.0],
                "score": 0.9,
                "text": "ABC1234",
                "plate_color": "blue",
                "plate_color_conf": 0.88,
                "plate_type": "single",
                "landmarks": [[1, 2], [11, 2], [11, 12], [1, 12]],
            }
        ]


class _CountingDualLpr:
    def __init__(self):
        self.calls = 0

    def infer_frame(self, frame, conf_thresh, iou_thresh):
        self.calls += 1
        return []


class WorkerPlateColorConfTests(unittest.TestCase):
    def _build_worker(
        self,
        rk=None,
        no_draw=True,
        draw_plate_boxes=False,
        postprocessor=None,
        dual_lpr=None,
        plate_requires_vehicle=False,
    ):
        worker = DetectWorker.__new__(DetectWorker)
        worker.idx = 0
        worker.args = SimpleNamespace(
            no_draw=no_draw,
            draw_plate_boxes=draw_plate_boxes,
            conf=0.25,
            iou=0.5,
        )
        worker.core_mask = None
        worker.task_q = queue.Queue()
        worker.result_q = queue.Queue()
        worker.detect_mask = None
        worker.rk = rk or _FakeRK()
        worker.detector_postprocessor = postprocessor or _FakePostprocessor()
        worker.dual_lpr = dual_lpr or _FakeDualLpr()
        worker.plate_infer_stride = 1
        worker.plate_requires_vehicle = plate_requires_vehicle
        worker.frames = 0
        worker.infer_time = 0.0
        return worker

    def test_worker_passthrough_plate_color_conf_to_det_payload(self):
        worker = self._build_worker()

        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((0, frame))
        worker.task_q.put(None)

        worker.run()

        result = worker.result_q.get_nowait()
        self.assertIsNotNone(result)
        _, _, _, _, det_payload = result
        plate_items = [item for item in det_payload if int(item.get("cls", -1)) == int(LICENSE_CLASS)]
        self.assertEqual(len(plate_items), 1)
        self.assertAlmostEqual(float(plate_items[0]["plate_color_conf"]), 0.88, places=6)

    def test_worker_plate_drawing_requires_explicit_flag(self):
        worker = self._build_worker(no_draw=False, draw_plate_boxes=False)

        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((0, frame))
        worker.task_q.put(None)

        worker.run()

        _, _, frame_out, _, _ = worker.result_q.get_nowait()
        self.assertTrue(np.array_equal(frame_out, frame))

    def test_worker_draws_plate_when_explicitly_enabled(self):
        worker = self._build_worker(no_draw=False, draw_plate_boxes=True)

        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((0, frame))
        worker.task_q.put(None)

        worker.run()

        _, _, frame_out, _, _ = worker.result_q.get_nowait()
        self.assertFalse(np.array_equal(frame_out, frame))

    def test_worker_reports_empty_result_and_finishes_queue_when_frame_fails(self):
        worker = self._build_worker(rk=_FailingRK())
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((12, frame, 123.0))
        worker.task_q.put(None)

        worker.run()

        result = worker.result_q.get_nowait()
        self.assertIsNotNone(result)
        frame_idx, capture_ts, frame_out, rows, det_payload = result
        self.assertEqual(frame_idx, 12)
        self.assertEqual(capture_ts, 123.0)
        self.assertIs(frame_out, frame)
        self.assertEqual(rows, [])
        self.assertEqual(det_payload, [])
        self.assertIsNone(worker.result_q.get_nowait())
        self.assertEqual(worker.task_q.unfinished_tasks, 0)

    def test_worker_skips_plate_inference_when_no_vehicle_candidates(self):
        dual_lpr = _CountingDualLpr()
        worker = self._build_worker(
            postprocessor=_FakeWaterOnlyPostprocessor(),
            dual_lpr=dual_lpr,
            plate_requires_vehicle=True,
        )
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((3, frame))
        worker.task_q.put(None)

        worker.run()

        self.assertEqual(dual_lpr.calls, 0)
        result = worker.result_q.get_nowait()
        self.assertIsNotNone(result)
        _, _, _, _, det_payload = result
        plate_items = [item for item in det_payload if int(item.get("cls", -1)) == int(LICENSE_CLASS)]
        self.assertEqual(plate_items, [])

    def test_worker_runs_plate_inference_without_vehicle_when_override_disabled(self):
        dual_lpr = _CountingDualLpr()
        worker = self._build_worker(
            postprocessor=_FakeWaterOnlyPostprocessor(),
            dual_lpr=dual_lpr,
            plate_requires_vehicle=False,
        )
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((4, frame))
        worker.task_q.put(None)

        worker.run()

        self.assertEqual(dual_lpr.calls, 1)

    def test_worker_runs_plate_inference_without_main_detections_when_vehicle_gate_disabled(self):
        worker = self._build_worker(
            postprocessor=_FakeNoDetectionsPostprocessor(),
            dual_lpr=_FakeDualLpr(),
            plate_requires_vehicle=False,
        )
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((5, frame))
        worker.task_q.put(None)

        worker.run()

        result = worker.result_q.get_nowait()
        self.assertIsNotNone(result)
        _, _, _, _, det_payload = result
        plate_items = [item for item in det_payload if int(item.get("cls", -1)) == int(LICENSE_CLASS)]
        self.assertEqual(len(plate_items), 1)

    def test_worker_skips_plate_inference_without_main_detections_when_vehicle_gate_enabled(self):
        dual_lpr = _CountingDualLpr()
        worker = self._build_worker(
            postprocessor=_FakeNoDetectionsPostprocessor(),
            dual_lpr=dual_lpr,
            plate_requires_vehicle=True,
        )
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        worker.task_q.put((6, frame))
        worker.task_q.put(None)

        worker.run()

        self.assertEqual(dual_lpr.calls, 0)


if __name__ == "__main__":
    unittest.main()
