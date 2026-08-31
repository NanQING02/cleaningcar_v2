from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from .constants import PLATE_COLOR_NAMES, PLATE_DECODE_CHARS
from .resize_accel import resize_bgr

_PLATE_NAME = PLATE_DECODE_CHARS
_PLATE_COLORS = PLATE_COLOR_NAMES
_RAW_HEAD_ANCHORS = (
    np.array([[4, 5], [8, 10], [13, 16]], dtype=np.float32),
    np.array([[23, 29], [43, 55], [73, 105]], dtype=np.float32),
    np.array([[146, 217], [231, 300], [335, 433]], dtype=np.float32),
)


def _is_viable_plate_roi(img: np.ndarray, min_width: int = 8, min_height: int = 4) -> bool:
    if img is None or not isinstance(img, np.ndarray):
        return False
    if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        return False
    h, w = img.shape[:2]
    return h >= int(min_height) and w >= int(min_width)


def _letter_box(img: np.ndarray, size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)
    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    resized = resize_bgr(img, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, left, top


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, iou_thresh: float) -> List[int]:
    order = np.argsort(boxes[:, 4])[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (
            (boxes[order[1:], 2] - boxes[order[1:], 0])
            * (boxes[order[1:], 3] - boxes[order[1:], 1])
        )
        iou = inter / (area_i + area_o - inter + 1e-6)
        idx = np.where(iou <= iou_thresh)[0]
        order = order[idx + 1]
    return keep


def _restore_box(boxes: np.ndarray, r: float, left: int, top: int) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2, 5, 7, 9, 11]] -= left
    out[:, [1, 3, 6, 8, 10, 12]] -= top
    out[:, [0, 2, 5, 7, 9, 11]] /= r
    out[:, [1, 3, 6, 8, 10, 12]] /= r
    return out


def _decode_int8_detection_heads(outputs, conf_thresh: float, iou_thresh: float,
                                 scale: float, left: int, top: int) -> np.ndarray:
    candidates = []
    for raw_head, anchors in zip(outputs, _RAW_HEAD_ANCHORS):
        raw_head = np.asarray(raw_head, dtype=np.float32)
        if raw_head.ndim != 4 or raw_head.shape[1] != 45:
            return np.empty((0, 14), dtype=np.float32)
        _, _, height, width = raw_head.shape
        head = raw_head.reshape(1, 3, 15, height, width)[0].transpose(0, 2, 3, 1)
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((grid_x, grid_y), axis=-1)[None]
        stride = 640.0 / width
        head[..., :5] = 1.0 / (1.0 + np.exp(-head[..., :5]))
        head[..., 13:] = 1.0 / (1.0 + np.exp(-head[..., 13:]))
        scores = head[..., 4] * np.max(head[..., 13:], axis=-1)
        anchor = anchors[:, None, None, :]
        centers = (head[..., :2] * 2.0 - 0.5 + grid) * stride
        sizes = (head[..., 2:4] * 2.0) ** 2 * anchor
        landmarks = head[..., 5:13].reshape(3, height, width, 4, 2)
        landmarks = landmarks * anchor[..., None, :] + grid[..., None, :] * stride
        for anchor_index, row, column in zip(*np.where(scores > float(conf_thresh))):
            center_x, center_y = centers[anchor_index, row, column]
            box_width, box_height = sizes[anchor_index, row, column]
            candidates.append([
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
                scores[anchor_index, row, column],
                *landmarks[anchor_index, row, column].reshape(-1),
                np.argmax(head[anchor_index, row, column, 13:]),
            ])
    if not candidates:
        return np.empty((0, 14), dtype=np.float32)
    detections = np.asarray(candidates, dtype=np.float32)
    detections = detections[_nms(detections, float(iou_thresh))]
    return _restore_box(detections, scale, left, top).astype(np.float32)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts.astype(np.float32))
    tl, tr, br, bl = rect
    width_a = float(np.hypot(br[0] - bl[0], br[1] - bl[1]))
    width_b = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
    max_w = max(int(width_a), int(width_b))
    height_a = float(np.hypot(tr[0] - br[0], tr[1] - br[1]))
    height_b = float(np.hypot(tl[0] - bl[0], tl[1] - bl[1]))
    max_h = max(int(height_a), int(height_b))
    if max_w <= 1 or max_h <= 1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_w, max_h))


def _split_merge_double_plate(img: np.ndarray) -> np.ndarray:
    h, _ = img.shape[:2]
    if h < 6:
        return img
    upper = img[0 : int(5.0 / 12.0 * h), :]
    lower = img[int(1.0 / 3.0 * h) :, :]
    if lower.size == 0:
        return img
    upper = resize_bgr(upper, (lower.shape[1], lower.shape[0]))
    return np.hstack((upper, lower))


def _decode_plate(indices: Sequence[int]) -> str:
    prev = 0
    out: List[str] = []
    for v in indices:
        idx = int(v)
        if idx != 0 and idx != prev and 0 <= idx < len(_PLATE_NAME):
            out.append(_PLATE_NAME[idx])
        prev = idx
    return "".join(out)


def _decode_plate_with_confidence(indices: Sequence[int], confidences: Sequence[float]) -> Tuple[str, float]:
    prev = 0
    chars = []
    selected_confidences = []
    for value, confidence in zip(indices, confidences):
        index = int(value)
        if index != 0 and index != prev and 0 <= index < len(_PLATE_NAME):
            chars.append(_PLATE_NAME[index])
            selected_confidences.append(float(confidence))
        prev = index
    if not selected_confidences:
        return '', 0.0
    return ''.join(chars), float(sum(selected_confidences) / len(selected_confidences))


class DualPlateRecognizer:
    def __init__(
        self,
        detect_model_path: str,
        rec_model_path: str,
        verbose: bool = False,
        core_mask=None,
        log_output_shape_once: bool = True,
    ):
        self.verbose = bool(verbose)
        self.log_output_shape_once = bool(log_output_shape_once)
        self._detect_shape_logged = False
        self._rec_shape_logged = False
        self._detect_shape_warned = False
        self._rec_shape_warned = False
        self.detector = RKNNLite(verbose=self.verbose)
        self.recognizer = RKNNLite(verbose=self.verbose)

        detect_path = Path(detect_model_path)
        rec_path = Path(rec_model_path)
        if not detect_path.exists():
            raise FileNotFoundError(f"detect model not found: {detect_path}")
        if not rec_path.exists():
            raise FileNotFoundError(f"rec model not found: {rec_path}")

        if self.detector.load_rknn(str(detect_path)) != 0:
            raise RuntimeError(f"failed to load detect model: {detect_path}")
        if self.recognizer.load_rknn(str(rec_path)) != 0:
            raise RuntimeError(f"failed to load rec model: {rec_path}")
        init_kwargs = {}
        if core_mask is not None:
            init_kwargs["core_mask"] = core_mask
        if self.detector.init_runtime(**init_kwargs) != 0:
            raise RuntimeError("failed to init detect runtime")
        if self.recognizer.init_runtime(**init_kwargs) != 0:
            raise RuntimeError("failed to init rec runtime")

    def release(self) -> None:
        try:
            self.detector.release()
        except Exception:
            pass
        try:
            self.recognizer.release()
        except Exception:
            pass

    def __del__(self) -> None:
        self.release()

    @staticmethod
    def _shape_list(outputs) -> List[Any]:
        if not outputs:
            return []
        shapes = []
        for item in outputs:
            shape = getattr(item, "shape", None)
            if shape is None:
                shapes.append(None)
            else:
                shapes.append(tuple(int(v) for v in shape))
        return shapes

    def _log_shapes_once(self, kind: str, outputs) -> None:
        if not getattr(self, "log_output_shape_once", True):
            return
        if kind == "detect":
            if getattr(self, "_detect_shape_logged", False):
                return
            self._detect_shape_logged = True
        elif kind == "recognize":
            if getattr(self, "_rec_shape_logged", False):
                return
            self._rec_shape_logged = True
        else:
            return
        print(f"[dual-plate] {kind} output shapes: {self._shape_list(outputs)}")

    def _warn_shape_once(self, kind: str, message: str) -> None:
        if kind == "detect":
            if getattr(self, "_detect_shape_warned", False):
                return
            self._detect_shape_warned = True
        elif kind == "recognize":
            if getattr(self, "_rec_shape_warned", False):
                return
            self._rec_shape_warned = True
        else:
            return
        print(f"[dual-plate] {kind} output shape warning: {message}")

    def _detect(self, frame_bgr: np.ndarray, conf_thresh: float, iou_thresh: float) -> np.ndarray:
        img_letterbox, r, left, top = _letter_box(frame_bgr, (640, 640))
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
        outputs = self.detector.inference(inputs=[inp], data_format=["nhwc"])
        self._log_shapes_once("detect", outputs)
        if not outputs:
            return np.empty((0, 14), dtype=np.float32)

        if len(outputs) == 3 and all(np.asarray(item).ndim == 4 for item in outputs):
            expected_channels = 45
            invalid_heads = [
                tuple(np.asarray(item).shape) for item in outputs
                if np.asarray(item).shape[0] != 1 or np.asarray(item).shape[1] != expected_channels
            ]
            if invalid_heads:
                self._warn_shape_once(
                    'detect',
                    f'unsupported INT8 raw-head schema, expected [1,{expected_channels},H,W], got {invalid_heads}',
                )
                return np.empty((0, 14), dtype=np.float32)
            return _decode_int8_detection_heads(outputs, conf_thresh, iou_thresh, r, left, top)

        dets = outputs[0]
        if dets is None:
            return np.empty((0, 14), dtype=np.float32)
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim == 3:
            dets = dets[0]
        if dets.ndim != 2 or dets.shape[1] < 15:
            self._warn_shape_once(
                "detect",
                f"expected 2D output with at least 15 columns, got shape={getattr(dets, 'shape', None)}",
            )
            return np.empty((0, 14), dtype=np.float32)

        dets = dets[dets[:, 4] > float(conf_thresh)]
        if dets.shape[0] == 0:
            return np.empty((0, 14), dtype=np.float32)

        dets[:, 13:15] *= dets[:, 4:5]
        boxes = _xywh2xyxy(dets[:, :4])
        scores = np.max(dets[:, 13:15], axis=-1, keepdims=True)
        cls_idx = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
        out = np.concatenate((boxes, scores, dets[:, 5:13], cls_idx), axis=1)
        out = out[_nms(out, float(iou_thresh))]
        out = _restore_box(out, r, left, top)
        return out.astype(np.float32)

    @staticmethod
    def _softmax_1d(logits: np.ndarray) -> np.ndarray:
        vec = np.asarray(logits, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return np.empty((0,), dtype=np.float32)
        shifted = vec - float(np.max(vec))
        exp_v = np.exp(shifted).astype(np.float32)
        denom = float(np.sum(exp_v))
        if denom <= 1e-12:
            return np.zeros_like(vec, dtype=np.float32)
        return (exp_v / denom).astype(np.float32)

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted).astype(np.float32)
        return exp_values / np.maximum(np.sum(exp_values, axis=1, keepdims=True), 1e-12)

    def _recognize(self, plate_bgr: np.ndarray) -> Tuple[str, str, float, float]:
        if not _is_viable_plate_roi(plate_bgr):
            return "", "", 0.0, 0.0

        plate = resize_bgr(plate_bgr, (168, 48))
        inp = np.expand_dims(plate, axis=0).astype(np.uint8)
        outputs = self.recognizer.inference(inputs=[inp], data_format=["nhwc"])
        self._log_shapes_once("recognize", outputs)
        if not outputs or len(outputs) < 2:
            self._warn_shape_once("recognize", f"expected plate logits and color logits, got {self._shape_list(outputs)}")
            return "", "", 0.0, 0.0

        plate_logits = outputs[0]
        color_logits = outputs[1]
        if plate_logits is None or color_logits is None:
            self._warn_shape_once("recognize", f"got None output: {self._shape_list(outputs)}")
            return "", "", 0.0, 0.0

        plate_logits = np.asarray(plate_logits, dtype=np.float32)
        color_logits = np.asarray(color_logits, dtype=np.float32)
        if plate_logits.ndim == 3:
            plate_logits = plate_logits[0]
        if plate_logits.ndim != 2:
            self._warn_shape_once("recognize", f"plate logits must be 2D after batch squeeze, got {plate_logits.shape}")
            return "", "", 0.0, 0.0

        if plate_logits.shape[1] == len(_PLATE_NAME):
            normalized_logits = plate_logits
        elif plate_logits.shape[0] == len(_PLATE_NAME):
            normalized_logits = plate_logits.transpose(1, 0)
        else:
            self._warn_shape_once(
                "recognize",
                f"plate logits must be [T,C] or [C,T] with C={len(_PLATE_NAME)}, got {plate_logits.shape}",
            )
            return "", "", 0.0, 0.0

        token_probabilities = self._softmax_rows(normalized_logits)
        token_ids = np.argmax(token_probabilities, axis=1)
        token_confidences = np.max(token_probabilities, axis=1)
        text, text_confidence = _decode_plate_with_confidence(token_ids, token_confidences)
        color_probs = self._softmax_1d(color_logits)
        if color_probs.size == 0:
            return text, "", 0.0, text_confidence
        color_idx = int(np.argmax(color_probs))
        plate_color = _PLATE_COLORS[color_idx] if 0 <= color_idx < len(_PLATE_COLORS) else ""
        plate_color_conf = float(color_probs[color_idx]) if 0 <= color_idx < color_probs.size else 0.0
        return text, plate_color, plate_color_conf, text_confidence

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.5,
    ) -> List[Dict[str, Any]]:
        dets = self._detect(frame_bgr, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        results: List[Dict[str, Any]] = []

        for det in dets:
            box = det[:4].tolist()
            landmarks_np = det[5:13].reshape(4, 2)
            score = float(det[4])
            plate_type_id = int(det[-1])
            plate_type = "double" if plate_type_id == 1 else "single"

            roi = _four_point_transform(frame_bgr, landmarks_np)
            if not _is_viable_plate_roi(roi):
                text, plate_color, plate_color_conf, plate_text_conf = "", "", 0.0, 0.0
            else:
                if plate_type_id == 1:
                    roi = _split_merge_double_plate(roi)
                if not _is_viable_plate_roi(roi):
                    text, plate_color, plate_color_conf, plate_text_conf = "", "", 0.0, 0.0
                else:
                    text, plate_color, plate_color_conf, plate_text_conf = self._recognize(roi)

            results.append(
                {
                    "box": box,
                    "landmarks": landmarks_np.tolist(),
                    "text": text or "",
                    "plate_text_conf": float(plate_text_conf),
                    "plate_color": plate_color or "",
                    "plate_color_conf": float(plate_color_conf),
                    "plate_type": plate_type,
                    "score": score,
                }
            )

        return results


__all__ = ["DualPlateRecognizer"]
