from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .resize_accel import resize_bgr


@dataclass
class LetterboxInfo:
    orig_h: int
    orig_w: int
    input_h: int
    input_w: int
    scale: float
    dw: float
    dh: float


def letterbox(image: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(0, 0, 0)):
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        image = resize_bgr(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    info = LetterboxInfo(
        orig_h=int(shape[0]),
        orig_w=int(shape[1]),
        input_h=int(new_shape[0]),
        input_w=int(new_shape[1]),
        scale=float(scale),
        dw=float(dw),
        dh=float(dh),
    )
    return image, info


def _softmax(x: np.ndarray, axis: int):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def _dfl(position: np.ndarray):
    # position: [N, C, H, W], C = 4 * bins
    n, c, h, w = position.shape
    if c % 4 != 0:
        raise ValueError(f"DFL channels must be divisible by 4, got {c}")
    bins = c // 4
    y = position.reshape(n, 4, bins, h, w)
    y = _softmax(y, axis=2)
    acc = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    y = (y * acc).sum(axis=2)
    return y


def _box_process(position: np.ndarray, img_size: Tuple[int, int]):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    grid = np.concatenate((col, row), axis=1)

    # Match the validated FP RKNN model output layout and stride mapping.
    stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w], dtype=np.float32).reshape(1, 2, 1, 1)

    position = _dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def _flatten_spatial(values: np.ndarray):
    channels = values.shape[1]
    values = values.transpose(0, 2, 3, 1)
    return values.reshape(-1, channels)


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, nms_thresh: float):
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        index = order[0]
        keep.append(index)

        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w * h
        iou = inter / (areas[index] + areas[order[1:]] - inter)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def deletterbox_boxes(boxes: Optional[np.ndarray], info: LetterboxInfo):
    if boxes is None or len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, 0] = np.clip((out[:, 0] - info.dw) / info.scale, 0, info.orig_w)
    out[:, 1] = np.clip((out[:, 1] - info.dh) / info.scale, 0, info.orig_h)
    out[:, 2] = np.clip((out[:, 2] - info.dw) / info.scale, 0, info.orig_w)
    out[:, 3] = np.clip((out[:, 3] - info.dh) / info.scale, 0, info.orig_h)
    return out


class FpModelPostprocessor:
    """
    Reusable postprocessor for the active FP detection model:
    - black padding letterbox (pad=0)
    - compatible with both 6-output and 9-output RKNN layouts
    - DFL decode + class-wise NMS
    """

    needs_black_padding = True

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        obj_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        output_mode: str = "6",
        num_classes: int = 9,
    ):
        self.img_size = tuple(img_size)
        self.obj_thresh = float(obj_thresh)
        self.nms_thresh = float(nms_thresh)
        self.num_classes = int(num_classes)
        self.output_mode = self._normalize_output_mode(output_mode)

    @staticmethod
    def _normalize_output_mode(output_mode: str) -> str:
        text = str(output_mode or "6").strip().lower()
        aliases = {
            "6": "6",
            "box_class": "6",
            "box+class": "6",
            "9": "9",
            "box_class_score": "9",
            "box+class+score": "9",
        }
        if text not in aliases:
            raise ValueError(f"Unsupported fp output mode: {output_mode}")
        return aliases[text]

    def describe_mode(self) -> str:
        if self.output_mode == "9":
            return "9-output(box+class+score)"
        return "6-output(box+class)"

    def _classify_output_branch(self, arr: np.ndarray) -> str:
        channels = int(arr.shape[1])
        if channels == self.num_classes:
            return "class"
        if channels == 1:
            return "score"
        if channels % 4 == 0 and channels >= 16:
            return "box"
        raise ValueError(f"Unsupported output branch channels: {arr.shape}")

    def _group_outputs(self, outputs: Sequence[np.ndarray]):
        outs = []
        for idx, item in enumerate(outputs):
            arr = np.asarray(item)
            if arr.ndim != 4:
                raise ValueError(f"Output[{idx}] must be 4D [N,C,H,W], got {arr.shape}")
            outs.append(arr)

        groups: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        for arr in outs:
            key = (int(arr.shape[2]), int(arr.shape[3]))
            branch_name = self._classify_output_branch(arr)
            branch_group = groups.setdefault(key, {})
            if branch_name in branch_group:
                raise ValueError(f"Duplicate {branch_name} branch for spatial size {key}")
            branch_group[branch_name] = arr

        scales = []
        for key in sorted(groups.keys(), key=lambda item: item[0] * item[1], reverse=True):
            branch_group = groups[key]
            if "box" not in branch_group or "class" not in branch_group:
                raise ValueError(f"Incomplete fp outputs for spatial size {key}: {sorted(branch_group.keys())}")
            if self.output_mode == "9" and "score" not in branch_group:
                raise ValueError(f"fp output mode=9 requires score branch for spatial size {key}")
            scales.append(branch_group)
        return scales

    @staticmethod
    def _normalize_score_branch(score_values: np.ndarray):
        score_values = score_values.astype(np.float32).reshape(-1)
        if score_values.size == 0:
            return score_values
        score_min = float(np.min(score_values))
        score_max = float(np.max(score_values))
        if score_min < 0.0 or score_max > 1.0:
            return _sigmoid(score_values)
        return np.clip(score_values, 0.0, 1.0)

    def prepare(self, frame: np.ndarray):
        """
        Preprocess helper for caller:
        returns (nhwc_rgb_input, letterbox_info)
        """
        image, info = letterbox(frame, self.img_size, color=(0, 0, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[None, ...]  # [1, H, W, C]
        return image, info

    def postprocess(self, outputs: Sequence[np.ndarray]):
        """
        Consume RKNNLite.inference() outputs and return:
        boxes, classes, scores (letterbox coordinate space).
        """
        if len(outputs) not in (6, 9):
            raise ValueError(f"Expected 6 or 9 outputs from fp model, got {len(outputs)}")

        boxes = []
        class_confidences = []
        score_confidences = []
        for branch_group in self._group_outputs(outputs):
            boxes.append(_box_process(branch_group["box"], self.img_size))
            class_confidences.append(branch_group["class"])
            if self.output_mode == "9":
                score_confidences.append(branch_group["score"])

        boxes = np.concatenate([_flatten_spatial(item) for item in boxes], axis=0)
        class_confidences = np.concatenate([_flatten_spatial(item) for item in class_confidences], axis=0)
        if self.output_mode == "9":
            score_values = np.concatenate([_flatten_spatial(item) for item in score_confidences], axis=0)
            score_values = self._normalize_score_branch(score_values.reshape(-1))
            if score_values.shape[0] != class_confidences.shape[0]:
                raise ValueError(
                    f"Score branch shape mismatch: score={score_values.shape[0]} class={class_confidences.shape[0]}"
                )
            class_confidences = class_confidences * score_values[:, None]

        scores = np.max(class_confidences, axis=-1)
        classes = np.argmax(class_confidences, axis=-1)

        keep = np.where(scores >= self.obj_thresh)[0]
        if keep.size == 0:
            return None, None, None

        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]

        kept_boxes = []
        kept_classes = []
        kept_scores = []
        for class_id in sorted(set(classes.tolist())):
            inds = np.where(classes == class_id)[0]
            class_boxes = boxes[inds]
            class_scores = scores[inds]
            keep_inds = _nms_boxes(class_boxes, class_scores, self.nms_thresh)
            if len(keep_inds):
                kept_boxes.append(class_boxes[keep_inds])
                kept_classes.append(classes[inds][keep_inds])
                kept_scores.append(class_scores[keep_inds])

        if not kept_boxes:
            return None, None, None

        return np.concatenate(kept_boxes), np.concatenate(kept_classes), np.concatenate(kept_scores)

    def map_boxes_to_original(self, boxes: Optional[np.ndarray], info: LetterboxInfo):
        """
        Map letterbox-space boxes back to original frame coordinates.
        """
        return deletterbox_boxes(boxes, info)
