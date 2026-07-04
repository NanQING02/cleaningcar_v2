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


class DualPlateRecognizer:
    def __init__(self, detect_model_path: str, rec_model_path: str, verbose: bool = False, core_mask=None):
        self.verbose = bool(verbose)
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

    def _detect(self, frame_bgr: np.ndarray, conf_thresh: float, iou_thresh: float) -> np.ndarray:
        img_letterbox, r, left, top = _letter_box(frame_bgr, (640, 640))
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
        outputs = self.detector.inference(inputs=[inp], data_format=["nhwc"])
        if not outputs:
            return np.empty((0, 14), dtype=np.float32)

        dets = outputs[0]
        if dets is None:
            return np.empty((0, 14), dtype=np.float32)
        if dets.ndim == 3:
            dets = dets[0]
        if dets.ndim != 2 or dets.shape[1] < 15:
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

    def _recognize(self, plate_bgr: np.ndarray) -> Tuple[str, str, float]:
        if not _is_viable_plate_roi(plate_bgr):
            return "", "", 0.0

        plate = resize_bgr(plate_bgr, (168, 48))
        inp = np.expand_dims(plate, axis=0).astype(np.uint8)
        outputs = self.recognizer.inference(inputs=[inp], data_format=["nhwc"])
        if not outputs or len(outputs) < 2:
            return "", "", 0.0

        plate_logits = outputs[0]
        color_logits = outputs[1]
        if plate_logits is None or color_logits is None:
            return "", "", 0.0

        if plate_logits.ndim == 3:
            plate_logits = plate_logits[0]
        if plate_logits.ndim != 2:
            return "", "", 0.0

        if plate_logits.shape[1] == len(_PLATE_NAME):
            token_ids = np.argmax(plate_logits, axis=1)
        elif plate_logits.shape[0] == len(_PLATE_NAME):
            token_ids = np.argmax(plate_logits, axis=0)
        else:
            token_ids = np.argmax(plate_logits, axis=1)

        text = _decode_plate(token_ids)
        color_probs = self._softmax_1d(color_logits)
        if color_probs.size == 0:
            return text, "", 0.0
        color_idx = int(np.argmax(color_probs))
        plate_color = _PLATE_COLORS[color_idx] if 0 <= color_idx < len(_PLATE_COLORS) else ""
        plate_color_conf = float(color_probs[color_idx]) if 0 <= color_idx < color_probs.size else 0.0
        return text, plate_color, plate_color_conf

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
                text, plate_color, plate_color_conf = "", "", 0.0
            else:
                if plate_type_id == 1:
                    roi = _split_merge_double_plate(roi)
                if not _is_viable_plate_roi(roi):
                    text, plate_color, plate_color_conf = "", "", 0.0
                else:
                    text, plate_color, plate_color_conf = self._recognize(roi)

            results.append(
                {
                    "box": box,
                    "landmarks": landmarks_np.tolist(),
                    "text": text or "",
                    "plate_color": plate_color or "",
                    "plate_color_conf": float(plate_color_conf),
                    "plate_type": plate_type,
                    "score": score,
                }
            )

        return results


__all__ = ["DualPlateRecognizer"]
