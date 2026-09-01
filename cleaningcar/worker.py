import threading
import time

import numpy as np
from rknnlite.api import RKNNLite

from .constants import (
    CLASS_NAMES,
    CLASS_THRESH,
    LICENSE_CLASS,
    VEHICLE_CLASS_IDS,
)
from .fp_detect import FpModelPostprocessor
from .plate_lpr import DualPlateRecognizer


class DetectWorker(threading.Thread):
    def __init__(self, idx, args, core_mask, task_q, result_q, plate_core_mask=None):
        super().__init__(daemon=True)
        self.idx = idx
        self.args = args
        self.core_mask = core_mask
        self.plate_core_mask = plate_core_mask
        self.task_q = task_q
        self.result_q = result_q
        self.rk = None
        self.dual_lpr = None
        logic_cfg = ((getattr(args, "_config", {}) or {}).get("logic", {}) or {})
        plate_output_shape_log_once = bool(logic_cfg.get("plate_output_shape_log_once", True))

        try:
            self.rk = RKNNLite()
            if self.rk.load_rknn(args.model) != 0:
                raise RuntimeError("load_rknn failed")
            init_kwargs = {}
            if core_mask is not None:
                init_kwargs["core_mask"] = core_mask
            if self.rk.init_runtime(**init_kwargs) != 0:
                raise RuntimeError("init_runtime failed")

            min_conf = min([float(args.conf)] + [float(v) for v in CLASS_THRESH.values()])
            self.detector_postprocessor = FpModelPostprocessor(
                img_size=(int(args.imgsz), int(args.imgsz)),
                obj_thresh=min_conf,
                nms_thresh=float(args.iou),
                output_mode=str(getattr(args, "fp_output_mode", "6")),
                num_classes=len(CLASS_NAMES),
            )
            print(f"Worker {self.idx}: fp_postprocess_mode={self.detector_postprocessor.describe_mode()}")
            self.dual_lpr = DualPlateRecognizer(
                detect_model_path=str(args.plate_detect_model),
                rec_model_path=str(args.plate_rec_model),
                verbose=False,
                core_mask=plate_core_mask,
                log_output_shape_once=plate_output_shape_log_once,
            )
        except Exception:
            self._release_runtime()
            raise
        self.plate_infer_stride = max(1, int(getattr(args, "plate_infer_stride", 1) or 1))
        explicit_plate_requires_vehicle = logic_cfg.get("plate_requires_vehicle")
        default_plate_requires_vehicle = bool(
            logic_cfg.get("disable_plate_only_events", False)
            or logic_cfg.get("require_vehicle_type_for_events", False)
        )
        self.plate_requires_vehicle = bool(
            getattr(
                args,
                "plate_requires_vehicle",
                default_plate_requires_vehicle if explicit_plate_requires_vehicle is None else explicit_plate_requires_vehicle,
            )
        )
        if self.idx == 0:
            print(
                f"plate_infer_stride={self.plate_infer_stride} "
                f"plate_core_mask={self.plate_core_mask} "
                f"plate_requires_vehicle={self.plate_requires_vehicle}"
            )

        self.frames = 0
        self.infer_time = 0.0

    def _release_runtime(self):
        dual_lpr = getattr(self, "dual_lpr", None)
        if dual_lpr is not None:
            release_fn = getattr(dual_lpr, "release", None)
            if callable(release_fn):
                try:
                    release_fn()
                except Exception:
                    pass
            self.dual_lpr = None

        rk = getattr(self, "rk", None)
        if rk is not None:
            release_fn = getattr(rk, "release", None)
            if callable(release_fn):
                try:
                    release_fn()
                except Exception:
                    pass
            self.rk = None

    def _put_empty_result(self, frame_idx, capture_ts, frame):
        self.result_q.put((frame_idx, capture_ts, frame, [], []))

    def run(self):
        while True:
            item = self.task_q.get()
            if item is None:
                self.task_q.task_done()
                break

            frame_idx = None
            frame = None
            capture_ts = None
            try:
                if len(item) == 2:
                    frame_idx, frame = item
                    capture_ts = None
                else:
                    frame_idx, frame, capture_ts = item

                img_input, lb_info = self.detector_postprocessor.prepare(frame)
                t0 = time.time()
                outputs = self.rk.inference(inputs=[img_input], data_format=["nhwc"])
                infer_time = time.time() - t0
                self.infer_time += infer_time
                self.frames += 1

                csv_rows = []
                det_payload = []
                base_frame = frame
                draw_frame = frame.copy()
                has_vehicle_candidates = False
                primary_boxes = np.empty((0, 4), dtype=np.float32)
                primary_scores = np.empty((0,), dtype=np.float32)
                primary_classes = np.empty((0,), dtype=np.int32)
                primary_plate_boxes = np.empty((0, 4), dtype=np.float32)
                primary_plate_scores = np.empty((0,), dtype=np.float32)

                if outputs:
                    boxes, classes, scores = self.detector_postprocessor.postprocess(outputs)
                    if boxes is not None and classes is not None and scores is not None:
                        boxes = self.detector_postprocessor.map_boxes_to_original(boxes, lb_info)
                        per_class_conf = np.array(
                            [CLASS_THRESH.get(int(c), self.args.conf) for c in classes],
                            dtype=np.float32,
                        )
                        keep = scores >= per_class_conf
                        if np.any(keep):
                            boxes = boxes[keep]
                            scores = scores[keep]
                            classes = classes[keep]
                            has_vehicle_candidates = bool(np.any(np.isin(classes, list(VEHICLE_CLASS_IDS))))
                            primary_keep = classes != LICENSE_CLASS
                            primary_boxes = boxes[primary_keep]
                            primary_scores = scores[primary_keep]
                            primary_classes = classes[primary_keep]
                            primary_plate_mask = classes == LICENSE_CLASS
                            primary_plate_boxes = boxes[primary_plate_mask]
                            primary_plate_scores = scores[primary_plate_mask]

                for box, score, cls_id in zip(primary_boxes, primary_scores, primary_classes):
                    x1, y1, x2, y2 = box.astype(int)

                    label_name = CLASS_NAMES[int(cls_id)]
                    csv_rows.append([frame_idx, label_name, f"{score:.4f}", x1, y1, x2, y2, -1, "", ""])
                    det_payload.append(
                        {
                            "cls": int(cls_id),
                            "score": float(score),
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "text": "",
                            "plate_color": "",
                            "plate_color_conf": None,
                            "plate_type": "",
                            "row_idx": len(csv_rows) - 1,
                            "label": label_name,
                        }
                    )

                for box, score in zip(primary_plate_boxes, primary_plate_scores):
                    x1, y1, x2, y2 = box.astype(int)
                    det_payload.append(
                        {
                            "cls": int(LICENSE_CLASS),
                            "score": float(score),
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "text": "",
                            "raw_text": "",
                            "plate_color": "",
                            "plate_color_conf": None,
                            "plate_type": "",
                            "source": "primary_plate_aux",
                            "row_idx": -1,
                            "label": CLASS_NAMES[LICENSE_CLASS],
                        }
                    )

                dual_plate_results = []
                should_run_plate = (frame_idx % self.plate_infer_stride == 0)
                if self.plate_requires_vehicle and not has_vehicle_candidates:
                    should_run_plate = False
                if should_run_plate:
                    try:
                        dual_plate_results = self.dual_lpr.infer_frame(
                            base_frame,
                            conf_thresh=min(float(self.args.conf), 0.3),
                            iou_thresh=float(self.args.iou),
                        )
                    except Exception as exc:
                        if frame_idx == 0 or frame_idx % 300 == 0:
                            print(f"Worker {self.idx}: dual plate inference failed: {exc}")

                for item in dual_plate_results:
                    box = item.get("box") or []
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    label_name = CLASS_NAMES[LICENSE_CLASS]
                    score = float(item.get("score", 0.0))
                    plate_text = str(item.get("text", "") or "")
                    raw_text_conf = item.get("plate_text_conf")
                    try:
                        plate_text_conf = float(raw_text_conf) if raw_text_conf is not None else None
                    except (TypeError, ValueError):
                        plate_text_conf = None
                    plate_color = str(item.get("plate_color", "") or "")
                    raw_color_conf = item.get("plate_color_conf")
                    try:
                        plate_color_conf = float(raw_color_conf) if raw_color_conf is not None else None
                    except (TypeError, ValueError):
                        plate_color_conf = None
                    plate_type = str(item.get("plate_type", "") or "")

                    csv_rows.append(
                        [frame_idx, label_name, f"{score:.4f}", x1, y1, x2, y2, -1, plate_text, plate_text]
                    )
                    det_payload.append(
                        {
                            "cls": int(LICENSE_CLASS),
                            "score": score,
                            "box": [x1, y1, x2, y2],
                            "text": plate_text,
                            "raw_text": plate_text,
                            "plate_text_conf": plate_text_conf,
                            "plate_color": plate_color,
                            "plate_color_conf": plate_color_conf,
                            "plate_type": plate_type,
                            "landmarks": item.get("landmarks"),
                            "source": "dual_plate",
                            "row_idx": len(csv_rows) - 1,
                            "label": label_name,
                        }
                    )

                self.result_q.put((frame_idx, capture_ts, draw_frame, csv_rows, det_payload))
            except Exception as exc:
                print(f"Worker {self.idx}: frame {frame_idx} failed: {exc}")
                if frame is not None:
                    self._put_empty_result(frame_idx, capture_ts, frame)
            finally:
                self.task_q.task_done()

        self._release_runtime()
        self.result_q.put(None)
