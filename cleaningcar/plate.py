from collections import Counter

from collections import Counter, deque

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


_BODY_CONFUSION_MAP = {
    "O": "0",
    "I": "1",
}


def normalize_plate_text(text):
    if not text:
        return ""
    text = text.upper().replace("路", "").replace(".", "").replace(" ", "")
    filtered = "".join(ch for ch in text if ch in PLATE_ALLOWED_CHARS)
    return filtered


def _normalize_plate_ocr_text(text):
    if not text:
        return ""
    text = text.upper().replace("路", "").replace(".", "").replace(" ", "")
    allowed = set(PLATE_ALLOWED_CHARS) | set(_BODY_CONFUSION_MAP.keys())
    return "".join(ch for ch in text if ch in allowed)


def _split_plate_tail(text):
    if text.endswith('险品'):
        return text[:-2], '险品'
    if text and text[-1] in PLATE_SUFFIX_CHARS:
        return text[:-1], text[-1]
    return text, ''


def _has_numeric_serial(text):
    body_with_prefix, _tail = _split_plate_tail(text)
    return any(ch in '0123456789' for ch in body_with_prefix[2:])


def _is_valid_plate_normalized(text):
    if not text:
        return False
    if len(text) < 7 or len(text) > 9:
        return False
    if text[0] not in PROVINCE_CHARS or text[1] not in PLATE_LETTERS:
        return False
    if not _has_numeric_serial(text):
        return False

    if PLATE_REGEX.match(text):
        return True
    if PLATE_REGEX_NE.match(text):
        return True

    body_with_prefix, tail = _split_plate_tail(text)
    if not tail:
        return False

    body = body_with_prefix[2:]
    if len(body) < 4 or len(body) > 6:
        return False
    if not all(ch in ALNUM for ch in body):
        return False
    return True


def normalize_plate_candidate_text(text):
    text = _normalize_plate_ocr_text(text)
    if len(text) >= 2 and not _has_numeric_serial(text):
        return text
    if not text or _is_valid_plate_normalized(text):
        body_with_prefix, tail = _split_plate_tail(text)
        chars = list(body_with_prefix)
        changed = False
        for idx in range(2, len(chars)):
            if chars[idx] == "O":
                chars[idx] = "0"
                changed = True
            elif chars[idx] == "I":
                chars[idx] = "1"
                changed = True
        if changed:
            corrected = ''.join(chars) + tail
            if _is_valid_plate_normalized(corrected):
                return corrected
        return text
    if len(text) < 3:
        return text

    body_with_prefix, tail = _split_plate_tail(text)
    chars = list(body_with_prefix)
    changed = False
    for idx in range(2, len(chars)):
        replacement = _BODY_CONFUSION_MAP.get(chars[idx])
        if replacement:
            chars[idx] = replacement
            changed = True
    if not changed:
        return text
    corrected = ''.join(chars) + tail
    if _is_valid_plate_normalized(corrected):
        return corrected
    return text


def is_valid_plate(text):
    text = normalize_plate_candidate_text(text)
    return _is_valid_plate_normalized(text)


class PlateVoteEvidence:
    """同一辆车的原始号码证据；按捕获时间投票，不依赖连续源帧。"""

    def __init__(self, history_seconds=6.0, max_observations=60, lifetime_max_candidates=32):
        self.history_seconds = max(0.1, float(history_seconds))
        self.observations = deque(maxlen=max(1, int(max_observations)))
        self.lifetime_max_candidates = max(2, int(lifetime_max_candidates))
        self.lifetime_counts = Counter()
        self.lifetime_first_frame = None
        self.lifetime_last_frame = None
        self.lifetime_ambiguous = False
        self.last_observed_frame = -1

    def expire(self, capture_ts):
        cutoff = float(capture_ts) - self.history_seconds
        while self.observations and self.observations[0]['capture_ts'] < cutoff:
            self.observations.popleft()

    def add(self, frame_idx, capture_ts, text, count_lifetime=True):
        text = normalize_plate_candidate_text(text)
        capture_ts = float(capture_ts)
        self.expire(capture_ts)
        if not is_valid_plate(text):
            return False
        frame_idx = int(frame_idx)
        # 同一源帧不能被多次update、缓存回放或handoff重复记票。
        if frame_idx <= self.last_observed_frame:
            return False
        if self.observations and capture_ts < self.observations[-1]['capture_ts']:
            return False
        self.observations.append({'frame_idx': frame_idx, 'capture_ts': capture_ts, 'text': text})
        self.last_observed_frame = frame_idx
        if count_lifetime:
            if self.lifetime_first_frame is None:
                self.lifetime_first_frame = frame_idx
            self.lifetime_last_frame = frame_idx
            if text in self.lifetime_counts or len(self.lifetime_counts) < self.lifetime_max_candidates:
                self.lifetime_counts[text] += 1
            else:
                # 不淘汰竞争号码来制造多数；超过有界容量就禁止收尾补锁。
                self.lifetime_ambiguous = True
        return True

    def ranked(self, capture_ts, window_seconds):
        self.expire(capture_ts)
        cutoff = float(capture_ts) - max(0.1, float(window_seconds))
        items = [item for item in self.observations if cutoff <= item['capture_ts'] <= capture_ts]
        counts = Counter(item['text'] for item in items)
        # 同票时没有多数，调用方不会锁定；保持排序确定，便于审计。
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return ranked, len(items)

    def majority(self, capture_ts, window_seconds, min_hits, min_ratio):
        ranked, total = self.ranked(capture_ts, window_seconds)
        if not ranked:
            return '', 0, 0
        text, hits = ranked[0]
        if hits >= int(min_hits) and hits / total >= float(min_ratio):
            return text, hits, total
        return '', hits, total

    def merge(self, other, capture_ts):
        """仅在业务层已经确认同车handoff之后合并，保留一份有界证据。"""
        if other is None or other is self:
            return
        if self.lifetime_first_frame is not None and other.lifetime_first_frame is not None:
            if max(self.lifetime_first_frame, other.lifetime_first_frame) <= min(
                self.lifetime_last_frame, other.lifetime_last_frame
            ):
                # 两个track的全程观测区间重叠，无法保证没有重复票或多车混入。
                self.lifetime_ambiguous = True
        self.lifetime_ambiguous = self.lifetime_ambiguous or other.lifetime_ambiguous
        for text, hits in other.lifetime_counts.items():
            if text in self.lifetime_counts or len(self.lifetime_counts) < self.lifetime_max_candidates:
                self.lifetime_counts[text] += hits
            else:
                self.lifetime_ambiguous = True
        first_frames = [f for f in (self.lifetime_first_frame, other.lifetime_first_frame) if f is not None]
        last_frames = [f for f in (self.lifetime_last_frame, other.lifetime_last_frame) if f is not None]
        self.lifetime_first_frame = min(first_frames) if first_frames else None
        self.lifetime_last_frame = max(last_frames) if last_frames else None
        items = {item['frame_idx']: dict(item) for item in other.observations}
        items.update({item['frame_idx']: dict(item) for item in self.observations})
        ordered = sorted(items.values(), key=lambda item: (item['capture_ts'], item['frame_idx']))
        self.observations.clear()
        self.observations.extend(ordered)
        self.expire(capture_ts)
        self.last_observed_frame = max(self.last_observed_frame, other.last_observed_frame)

    @staticmethod
    def _text_distance(left, right):
        previous = list(range(len(right) + 1))
        for i, a in enumerate(left, 1):
            row = [i]
            for j, b in enumerate(right, 1):
                row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
            previous = row
        return previous[-1]

    def finalize_candidate(self, min_hits=4, runner_up_ratio=2.0, identity_min_hits=2):
        ranked = sorted(self.lifetime_counts.items(), key=lambda item: (-item[1], item[0]))
        detail = {'candidates': ranked[:2], 'candidate_count': len(ranked), 'reason': ''}
        if self.lifetime_ambiguous:
            detail['reason'] = 'ambiguous_or_overflow'
        elif not ranked or ranked[0][1] < max(4, int(min_hits)):
            detail['reason'] = 'insufficient_votes'
        elif len(ranked) > 1 and ranked[0][1] < ranked[1][1] * max(2.0, float(runner_up_ratio)):
            detail['reason'] = 'competing_votes'
        elif any(hits >= max(2, int(identity_min_hits)) and self._text_distance(ranked[0][0], text) >= 3
                 for text, hits in ranked[1:]):
            detail['reason'] = 'distinct_plate_identity_conflict'
        else:
            detail['reason'] = 'lifetime_clear_winner'
            return ranked[0][0], detail
        return '', detail


class PlateTextTracker:
    def __init__(self, lock_frames=5, iou_thresh=0.4, max_age=30,
                 min_detection_confidence=0.65, min_recognition_confidence=0.75,
                 max_streak_gap_frames=2):
        self.lock_frames = max(1, lock_frames)
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_recognition_confidence = float(min_recognition_confidence)
        self.max_streak_gap_frames = max(1, int(max_streak_gap_frames))
        self.tracks = {}
        self.next_id = 1

    def _is_lock_eligible(self, det):
        text = normalize_plate_candidate_text(det.get('text', ''))
        if not is_valid_plate(text):
            return False
        try:
            detection_confidence = float(det.get('score', 0.0) or 0.0)
        except (TypeError, ValueError):
            detection_confidence = 0.0
        try:
            recognition_confidence = float(det.get('plate_text_conf', 0.0) or 0.0)
        except (TypeError, ValueError):
            recognition_confidence = 0.0
        return (
            detection_confidence >= self.min_detection_confidence
            and recognition_confidence >= self.min_recognition_confidence
        )

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
        det_texts = [normalize_plate_candidate_text(det.get("text", "")) for det in detections]
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
                        "streak_text": "",
                        "streak_hits": 0,
                        "streak_last_frame": -1,
                        "locked_text_conf": 0.0,
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
            if not track.get('locked') and self._is_lock_eligible(detections[det_idx]):
                track["history"].append(det_text)
                if len(track["history"]) > 30:
                    track["history"].pop(0)
                previous_text = track.get('streak_text', '')
                previous_frame = int(track.get('streak_last_frame', -1) or -1)
                if det_text == previous_text and frame_idx - previous_frame <= self.max_streak_gap_frames:
                    track['streak_hits'] = int(track.get('streak_hits', 0) or 0) + 1
                else:
                    track['streak_text'] = det_text
                    track['streak_hits'] = 1
                track['streak_last_frame'] = frame_idx
                if int(track.get('streak_hits', 0) or 0) >= self.lock_frames:
                    track['locked'] = det_text
                    track['locked_text_conf'] = float(detections[det_idx].get('plate_text_conf', 0.0) or 0.0)

        for di, det_box in enumerate(det_boxes):
            if di in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            det_text = det_texts[di]
            history = []
            if self._is_lock_eligible(detections[di]):
                history.append(det_text)
            self.tracks[tid] = {
                "box": det_box,
                "last_seen": frame_idx,
                "age": 0,
                "history": history,
                "streak_text": det_text if history else "",
                "streak_hits": 1 if history else 0,
                "streak_last_frame": frame_idx if history else -1,
                "locked_text_conf": 0.0,
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
                results.append({"track_id": -1, "text": normalize_plate_candidate_text(det.get("text", "")), "is_guess": False})
                continue
            track = self.tracks.get(tid)
            text = track.get("locked") or ""
            is_guess = False
            text_confidence = float(track.get('locked_text_conf', 0.0) or 0.0)
            if not text:
                raw_text = normalize_plate_candidate_text(det.get('text', ''))
                if is_valid_plate(raw_text):
                    text = raw_text
                    is_guess = True
                    try:
                        text_confidence = float(det.get('plate_text_conf', 0.0) or 0.0)
                    except (TypeError, ValueError):
                        text_confidence = 0.0
            results.append({
                "track_id": tid,
                "text": text,
                "is_guess": is_guess,
                "plate_text_conf": text_confidence,
            })
        return results
