import cv2
import numpy as np

from cleaningcar.constants import (
    CLAHE,
    LPR_BLANK,
    LPR_CHARS,
    PLATE_SIZE,
)


def extract_plate_patch(frame, box, expand_ratio):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1, 4)
    bh = max(y2 - y1, 4)
    bw *= 1.0 + expand_ratio
    bh *= 1.0 + expand_ratio * 0.5
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w - 1, int(cx + bw / 2))
    ny2 = min(h - 1, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return frame[ny1:ny2, nx1:nx2]


def enhance_plate_patch(patch):
    if patch is None or patch.size == 0:
        return None
    if patch.shape[0] > patch.shape[1] * 1.2:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    patch = cv2.resize(patch, PLATE_SIZE, interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    patch = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    patch = cv2.convertScaleAbs(patch, alpha=1.15, beta=5)
    return patch


def decode_lpr_output(pred):
    if pred is None:
        return ""
    arr = pred
    if arr.ndim == 3:
        if arr.shape[1] != len(LPR_CHARS):
            arr = np.transpose(arr, (0, 2, 1))
        arr = arr[0]
    elif arr.ndim != 2:
        return ""
    prev = LPR_BLANK
    chars = []
    for t in range(arr.shape[1]):
        c = int(np.argmax(arr[:, t]))
        if c == LPR_BLANK:
            prev = c
            continue
        if c != prev:
            chars.append(LPR_CHARS[c])
        prev = c
    return "".join(chars)
