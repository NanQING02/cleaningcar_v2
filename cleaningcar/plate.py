from collections import Counter

import numpy as np

from .constants import (
    ALNUM,
    PLATE_ALLOWED_CHARS,
    PLATE_LETTERS,
    PLATE_REGEX,
    PLATE_REGEX_NE,
    PROVINCE_CHARS,
    PLATE_SUFFIX_CHARS,
)


def normalize_plate_text(text):
    if not text:
        return ""
    text = text.upper().replace("路", "").replace(".", "").replace(" ", "")
    filtered = "".join(ch for ch in text if ch in PLATE_ALLOWED_CHARS)
    return filtered


def is_valid_plate(text):
    if not text:
        return False
    text = normalize_plate_text(text)
    if PLATE_REGEX.match(text):
        return True
    if PLATE_REGEX_NE.match(text):
        return True
    if len(text) < 7 or len(text) > 9:
        return False
    if text[0] not in PROVINCE_CHARS or text[1] not in PLATE_LETTERS:
        return False

    tail = ''
    if text.endswith('险品'):
        tail = '险品'
    elif text[-1] in PLATE_SUFFIX_CHARS:
        tail = text[-1]
    if not tail:
        return False

    body = text[2:-len(tail)]
    if len(body) < 4 or len(body) > 6:
        return False
    if not all(ch in ALNUM for ch in body):
        return False
    return True


class PlateTextTracker:
    def __init__(self, lock_frames=5, iou_thresh=0.4, max_age=30):
        self.lock_frames = max(1, lock_frames)
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        inter = interW * interH
        if inter <= 0:
            return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) + 1e-6
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) + 1e-6
        return inter / (areaA + areaB - inter)

    def update(self, frame_idx, detections):
        results = []
        det_boxes = [np.array(det["box"], dtype=float) for det in detections]
        det_texts = [normalize_plate_text(det.get("text", "")) for det in detections]
        track_ids = list(self.tracks.keys())
        iou_matrix = None
        if track_ids and det_boxes:
            iou_matrix = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tbox = self.tracks[tid]["box"]
                for di, dbox in enumerate(det_boxes):
                    iou_matrix[ti, di] = self._iou(tbox, dbox)

        assigned_tracks = {}
        assigned_dets = set()
        for det_idx, det in enumerate(detections):
            tid = det.get("track_id", -1)
            if tid and tid > 0:
                assigned_dets.add(det_idx)
                assigned_tracks[tid] = det_idx
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        "box": det_boxes[det_idx],
                        "last_seen": frame_idx,
                        "age": 0,
                        "history": [],
                        "locked": "",
                    }
                    if tid >= self.next_id:
                        self.next_id = tid + 1
                if iou_matrix is not None and tid in track_ids:
                    ti = track_ids.index(tid)
                    iou_matrix[ti, :] = -1
                    iou_matrix[:, det_idx] = -1

        if iou_matrix is not None:
            while True:
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[idx]
                if max_iou < self.iou_thresh:
                    break
                ti, di = idx
                tid = track_ids[ti]
                assigned_tracks[tid] = di
                assigned_dets.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        for tid, det_idx in assigned_tracks.items():
            det_box = det_boxes[det_idx]
            det_text = det_texts[det_idx]
            track = self.tracks[tid]
            track["box"] = det_box
            track["last_seen"] = frame_idx
            track["age"] = 0
            if is_valid_plate(det_text):
                track["history"].append(det_text)
                if len(track["history"]) > 30:
                    track["history"].pop(0)
                counts = Counter(track["history"])
                best_text, cnt = counts.most_common(1)[0]
                if cnt >= self.lock_frames:
                    track["locked"] = best_text

        for di, det_box in enumerate(det_boxes):
            if di in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            det_text = det_texts[di]
            history = []
            if is_valid_plate(det_text):
                history.append(det_text)
            self.tracks[tid] = {
                "box": det_box,
                "last_seen": frame_idx,
                "age": 0,
                "history": history,
                "locked": "",
            }
            assigned_tracks[tid] = di

        to_delete = []
        for tid, track in self.tracks.items():
            if tid in assigned_tracks:
                continue
            track["age"] += 1
            if track["age"] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)

        for det_idx, det in enumerate(detections):
            tid = None
            for track_id, d_idx in assigned_tracks.items():
                if d_idx == det_idx:
                    tid = track_id
                    break
            if tid is None:
                results.append({"track_id": -1, "text": normalize_plate_text(det.get("text", "")), "is_guess": False})
                continue
            track = self.tracks.get(tid)
            text = track.get("locked") or ""
            is_guess = False
            if not text:
                history = track.get("history") or []
                if history:
                    counts = Counter(history)
                    text, _ = counts.most_common(1)[0]
                    is_guess = True
            results.append({"track_id": tid, "text": text, "is_guess": is_guess})
        return results
