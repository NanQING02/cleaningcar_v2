from __future__ import annotations

import base64
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import cv2
import numpy as np

from .fp_detect import FpModelPostprocessor
from .log_throttle import WindowedLogThrottle
from .video_io import create_video_reader, parse_core_mask

DEFAULT_WHEEL_CLASSES = ["0-25", "25-50", "50-75", "75-100"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WHEEL_MODEL_PATH = (PROJECT_ROOT / "models" / "wheel" / "2026.4.28CRwheel.rknn").resolve()
DEFAULT_ENTRY_CLUSTER_SECONDS = 1.5
WHEEL_SIDES = ("left", "right")


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_wheel_model_path(model_value, base_dir=None):
    raw = str(model_value or "").strip()
    if not raw:
        return DEFAULT_WHEEL_MODEL_PATH
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir:
        from_base_dir = (Path(base_dir) / candidate).resolve()
        if from_base_dir.exists():
            return from_base_dir
    from_project_root = (PROJECT_ROOT / candidate).resolve()
    if from_project_root.exists():
        return from_project_root
    if base_dir:
        return (Path(base_dir) / candidate).resolve()
    return from_project_root


def resolve_wheel_class_name(class_names: Sequence[str], class_id: int) -> str:
    try:
        index = int(class_id)
    except (TypeError, ValueError):
        return ""
    if 0 <= index < len(class_names):
        return str(class_names[index])
    return str(index)


def resolve_wheel_settings(config, base_dir=None):
    raw = (config or {}).get("wheel", {}) or {}
    classes = raw.get("classes")
    if isinstance(classes, (list, tuple)):
        class_names = [str(item).strip() for item in classes if str(item).strip()]
    else:
        class_names = []
    if not class_names:
        class_names = list(DEFAULT_WHEEL_CLASSES)

    model_value = str(raw.get("model") or DEFAULT_WHEEL_MODEL_PATH)
    model_path = _resolve_wheel_model_path(model_value, base_dir=base_dir)

    target_fps = _safe_float(raw.get("target_fps", 10.0), 10.0)
    if target_fps <= 0.0:
        target_fps = 10.0

    active_target_fps = _safe_float(raw.get("active_target_fps", target_fps), target_fps)
    if active_target_fps < 0.0:
        active_target_fps = 0.0

    bind_window_seconds = _safe_float(raw.get("bind_window_seconds", 30.0), 30.0)
    if bind_window_seconds <= 0.0:
        bind_window_seconds = 30.0

    conf_thresh = _safe_float(raw.get("conf_thresh", 0.25), 0.25)
    if conf_thresh < 0.0:
        conf_thresh = 0.25

    nms_thresh = _safe_float(raw.get("nms_thresh", 0.45), 0.45)
    if nms_thresh <= 0.0:
        nms_thresh = 0.45
    imgsz = None
    if raw.get("imgsz") is not None:
        try:
            imgsz = max(64, int(raw.get("imgsz")))
        except (TypeError, ValueError):
            imgsz = None

    return {
        "enabled": _safe_bool(raw.get("enabled", False)),
        "event_driven": _safe_bool(raw.get("event_driven", True), True),
        "left_source": str(raw.get("left_source", "") or "").strip(),
        "right_source": str(raw.get("right_source", "") or "").strip(),
        "model": str(model_path),
        "model_path": Path(model_path),
        "classes": class_names,
        "target_fps": target_fps,
        "active_target_fps": active_target_fps,
        "bind_window_seconds": bind_window_seconds,
        "conf_thresh": conf_thresh,
        "nms_thresh": nms_thresh,
        "imgsz": imgsz,
        "core_mask": str(raw.get("core_mask", "") or "").strip(),
    }


def select_best_detection(
    boxes,
    classes,
    scores,
    frame_shape=None,
    class_names=None,
):
    if boxes is None or classes is None or scores is None:
        return None
    boxes = np.asarray(boxes)
    classes = np.asarray(classes)
    scores = np.asarray(scores)
    if boxes.size == 0 or classes.size == 0 or scores.size == 0:
        return None

    frame_h = 0
    frame_w = 0
    if frame_shape is not None and len(frame_shape) >= 2:
        frame_h = int(frame_shape[0])
        frame_w = int(frame_shape[1])
    frame_cx = frame_w * 0.5
    frame_cy = frame_h * 0.5

    class_names = list(class_names or DEFAULT_WHEEL_CLASSES)

    best_key = None
    best_item = None
    for box, cls_id, score in zip(boxes, classes, scores):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        score_f = float(score)
        cls_id_int = int(cls_id)
        candidate_key = (cls_id_int, -score_f)
        if best_key is not None and candidate_key >= best_key:
            continue
        best_key = candidate_key
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        distance = (center_x - frame_cx) ** 2 + (center_y - frame_cy) ** 2
        best_item = {
            "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "classId": cls_id_int,
            "score": score_f,
            "className": resolve_wheel_class_name(class_names, cls_id_int),
            "centerDistance": float(distance),
        }
    return best_item


def _encode_frame_jpeg_bytes(frame, image_quality=85):
    if frame is None:
        return b""
    quality = int(min(max(int(image_quality), 1), 100))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return encoded.tobytes()


def _jpeg_bytes_to_base64(data):
    if not data:
        return ""
    return base64.b64encode(data).decode("utf-8")


class WheelResultCache:
    def __init__(
        self,
        bind_window_seconds=30.0,
        image_quality=85,
        entry_cluster_seconds=DEFAULT_ENTRY_CLUSTER_SECONDS,
    ):
        self.bind_window_seconds = max(1.0, float(bind_window_seconds))
        self.image_quality = int(min(max(int(image_quality), 1), 100))
        self.entry_cluster_seconds = max(0.0, float(entry_cluster_seconds))
        self._entries: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._next_entry_id = 1

    def _prune_entries(self, side, cutoff_ts):
        entries = list(self._entries.get(side) or [])
        if not entries:
            self._entries.pop(side, None)
            return []
        kept = [entry for entry in entries if float(entry.get("capture_ts", 0.0) or 0.0) >= cutoff_ts]
        if kept:
            self._entries[side] = kept
        else:
            self._entries.pop(side, None)
        return kept

    @staticmethod
    def _entry_matches_time_window(entry, match_ref_ts, bind_window_seconds):
        capture_ts = float(entry.get("capture_ts", 0.0) or 0.0)
        if abs(float(match_ref_ts) - capture_ts) > float(bind_window_seconds):
            return False
        return True

    def update_from_detections(
        self,
        side,
        frame,
        capture_ts,
        boxes,
        classes,
        scores,
        class_names=None,
    ):
        side = str(side or "").strip().lower()
        if side not in WHEEL_SIDES or frame is None:
            return False
        candidate = select_best_detection(
            boxes=boxes,
            classes=classes,
            scores=scores,
            frame_shape=getattr(frame, "shape", ()),
            class_names=class_names or DEFAULT_WHEEL_CLASSES,
        )
        if not candidate:
            return False

        image_jpeg = _encode_frame_jpeg_bytes(frame, image_quality=self.image_quality)
        if not image_jpeg:
            return False

        capture_ts = time.time() if capture_ts is None else float(capture_ts)
        entry = {
            "entryId": int(self._next_entry_id),
            "side": side,
            "captureTime": datetime.fromtimestamp(capture_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "imageJpegBytes": image_jpeg,
            "className": candidate["className"],
            "score": float(candidate.get("score", 0.0) or 0.0),
            "centerDistance": float(candidate.get("centerDistance", 0.0) or 0.0),
            "capture_ts": capture_ts,
            "claimedTrackId": 0,
        }
        with self._lock:
            self._next_entry_id += 1
            cutoff_ts = capture_ts - self.bind_window_seconds
            entries = self._prune_entries(side, cutoff_ts)
            entries.append(entry)
            self._entries[side] = entries
        return True

    def get_recent_result_entries(self, now_ts=None, reference_ts=None, track_id=None):
        now_ref_ts = time.time() if now_ts is None else float(now_ts)
        match_ref_ts = now_ref_ts if reference_ts is None else float(reference_ts)
        track_id = int(track_id or 0)
        results = []
        with self._lock:
            for side in WHEEL_SIDES:
                entries = self._prune_entries(side, now_ref_ts - self.bind_window_seconds)
                if not entries:
                    continue
                candidates = []
                for entry in entries:
                    if not self._entry_matches_time_window(entry, match_ref_ts, self.bind_window_seconds):
                        continue
                    claimed_track_id = int(entry.get("claimedTrackId", 0) or 0)
                    if claimed_track_id > 0 and claimed_track_id != track_id:
                        continue
                    capture_ts = float(entry.get("capture_ts", 0.0) or 0.0)
                    time_delta = abs(match_ref_ts - capture_ts)
                    candidate_key = (
                        -float(entry.get("score", 0.0) or 0.0),
                        time_delta,
                        -capture_ts,
                    )
                    candidates.append((candidate_key, entry))
                if not candidates:
                    continue
                _, entry = min(candidates, key=lambda item: item[0])
                results.append(dict(entry))
        return results

    def get_claimed_result_entries(self, track_id, now_ts=None, reference_ts=None):
        now_ref_ts = time.time() if now_ts is None else float(now_ts)
        match_ref_ts = now_ref_ts if reference_ts is None else float(reference_ts)
        track_id = int(track_id or 0)
        if track_id <= 0:
            return []
        results = []
        with self._lock:
            for side in WHEEL_SIDES:
                entries = self._prune_entries(side, now_ref_ts - self.bind_window_seconds)
                for entry in entries:
                    if int(entry.get("claimedTrackId", 0) or 0) != track_id:
                        continue
                    if not self._entry_matches_time_window(entry, match_ref_ts, self.bind_window_seconds):
                        continue
                    results.append(dict(entry))
        results.sort(
            key=lambda item: (
                float(item.get("capture_ts", 0.0) or 0.0),
                int(item.get("entryId", 0) or 0),
            )
        )
        return results

    def get_photo_candidate_entries(self, track_id, now_ts=None, reference_ts=None):
        now_ref_ts = time.time() if now_ts is None else float(now_ts)
        match_ref_ts = now_ref_ts if reference_ts is None else float(reference_ts)
        track_id = int(track_id or 0)
        if track_id <= 0:
            return []
        near_window = self.entry_cluster_seconds if self.entry_cluster_seconds > 0.0 else self.bind_window_seconds
        results = []
        with self._lock:
            for side in WHEEL_SIDES:
                entries = self._prune_entries(side, now_ref_ts - self.bind_window_seconds)
                for entry in entries:
                    if not self._entry_matches_time_window(entry, match_ref_ts, self.bind_window_seconds):
                        continue
                    claimed_track_id = int(entry.get("claimedTrackId", 0) or 0)
                    if claimed_track_id == track_id:
                        results.append(dict(entry))
                        continue
                    if claimed_track_id > 0:
                        continue
                    capture_ts = float(entry.get("capture_ts", 0.0) or 0.0)
                    if abs(match_ref_ts - capture_ts) <= near_window:
                        results.append(dict(entry))
        results.sort(
            key=lambda item: (
                float(item.get("capture_ts", 0.0) or 0.0),
                int(item.get("entryId", 0) or 0),
            )
        )
        return results

    def claim_result_entry(self, track_id, entry_id):
        track_id = int(track_id or 0)
        entry_id = int(entry_id or 0)
        if track_id <= 0 or entry_id <= 0:
            return False
        with self._lock:
            for side in WHEEL_SIDES:
                entries = self._entries.get(side) or []
                for entry in entries:
                    if int(entry.get("entryId", 0) or 0) != entry_id:
                        continue
                    owner = int(entry.get("claimedTrackId", 0) or 0)
                    if owner > 0 and owner != track_id:
                        return False
                    ordered = sorted(
                        entries,
                        key=lambda item: float(item.get("capture_ts", 0.0) or 0.0),
                    )
                    target_idx = None
                    for idx, item in enumerate(ordered):
                        if int(item.get("entryId", 0) or 0) == entry_id:
                            target_idx = idx
                            break
                    if target_idx is None:
                        return False
                    cluster_ids = {entry_id}
                    if self.entry_cluster_seconds > 0.0:
                        left_idx = target_idx
                        while left_idx > 0:
                            curr_ts = float(ordered[left_idx].get("capture_ts", 0.0) or 0.0)
                            prev_ts = float(ordered[left_idx - 1].get("capture_ts", 0.0) or 0.0)
                            if abs(curr_ts - prev_ts) > self.entry_cluster_seconds:
                                break
                            cluster_ids.add(int(ordered[left_idx - 1].get("entryId", 0) or 0))
                            left_idx -= 1
                        right_idx = target_idx
                        while right_idx + 1 < len(ordered):
                            curr_ts = float(ordered[right_idx].get("capture_ts", 0.0) or 0.0)
                            next_ts = float(ordered[right_idx + 1].get("capture_ts", 0.0) or 0.0)
                            if abs(next_ts - curr_ts) > self.entry_cluster_seconds:
                                break
                            cluster_ids.add(int(ordered[right_idx + 1].get("entryId", 0) or 0))
                            right_idx += 1
                    entry["claimedTrackId"] = track_id
                    for sibling in entries:
                        sibling_id = int(sibling.get("entryId", 0) or 0)
                        if sibling_id not in cluster_ids:
                            continue
                        sibling_owner = int(sibling.get("claimedTrackId", 0) or 0)
                        if sibling_owner > 0 and sibling_owner != track_id:
                            continue
                        sibling["claimedTrackId"] = track_id
                    return True
        return False

    def get_recent_results(self, now_ts=None, reference_ts=None):
        results = []
        for entry in self.get_recent_result_entries(now_ts=now_ts, reference_ts=reference_ts):
            side = str(entry.get("side") or "").strip().lower()
            if side not in WHEEL_SIDES:
                continue
            results.append(
                {
                    "side": side,
                    "captureTime": str(entry.get("captureTime") or ""),
                    "imageBase64": _jpeg_bytes_to_base64(entry.get("imageJpegBytes", b"")),
                    "className": str(entry.get("className") or ""),
                }
            )
        return results

    def snapshot_stats(self, now_ts=None):
        now_ref_ts = time.time() if now_ts is None else float(now_ts)
        stats = {}
        with self._lock:
            for side in WHEEL_SIDES:
                entries = self._prune_entries(side, now_ref_ts - self.bind_window_seconds)
                newest = None
                if entries:
                    newest = max(entries, key=lambda item: float(item.get("capture_ts", 0.0) or 0.0))
                newest_ts = float(newest.get("capture_ts", 0.0) or 0.0) if newest else 0.0
                stats[side] = {
                    "entries": len(entries),
                    "claimed_entries": sum(
                        1 for item in entries if int(item.get("claimedTrackId", 0) or 0) > 0
                    ),
                    "newest_age": (now_ref_ts - newest_ts) if newest_ts > 0.0 else -1.0,
                    "newest_score": float(newest.get("score", 0.0) or 0.0) if newest else 0.0,
                    "newest_class": str(newest.get("className") or "") if newest else "",
                }
        return stats


class _LatestFrameSlot:
    def __init__(self):
        self._cond = threading.Condition()
        self._seq = 0
        self._item = None

    def put(self, item):
        with self._cond:
            self._seq += 1
            self._item = item
            self._cond.notify_all()

    def peek(self):
        with self._cond:
            return self._seq, self._item

    def wait_for_update(self, last_seq, timeout=0.2):
        deadline = None if timeout is None else (time.time() + max(0.0, float(timeout)))
        with self._cond:
            while self._seq <= last_seq:
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cond.wait(timeout=remaining)
            return self._seq, self._item


class WheelReaderThread(threading.Thread):
    def __init__(
        self,
        side,
        source,
        reader_args,
        frame_slot,
        stop_event,
        reader_fail_threshold=5,
        reconnect_delay=2.0,
    ):
        super().__init__(daemon=True)
        self.side = str(side)
        self.source = str(source)
        self.reader_args = reader_args
        self.frame_slot = frame_slot
        self.stop_event = stop_event
        self.reader_fail_threshold = max(1, int(reader_fail_threshold))
        self.reconnect_delay = max(0.2, float(reconnect_delay))
        self.frames = 0
        self.open_count = 0
        self.reconnect_count = 0
        self.frames_since_open = 0
        self.last_open_reason = "initial"
        self.last_reconnect_reason = ""
        self.last_frame_gap = -1.0
        self.last_open_age = -1.0
        self.last_open_delay = 0.0
        self.last_frame_ts = 0.0
        self._log_throttle = WindowedLogThrottle()

    def _log(self, key, message, window_seconds=10.0):
        self._log_throttle.log(key=key, message=message, window_seconds=window_seconds, emit=print)

    @staticmethod
    def _capture_failure_summary(cap):
        if cap is None or not hasattr(cap, "diagnostics"):
            return ""
        try:
            diag = cap.diagnostics() or {}
        except Exception:
            return ""
        parts = []
        last_error = str(diag.get("last_read_error") or "").strip()
        if last_error:
            parts.append(f"last_error={last_error}")
        recent_error_count = int(diag.get("recent_error_match_count") or 0)
        if recent_error_count:
            parts.append(f"recent_decode_errors={recent_error_count}")
        recent_error_lines = diag.get("recent_error_lines") or []
        if recent_error_lines:
            parts.append(f"error_tail={recent_error_lines[-4:]}")
        return " ".join(parts)

    def run(self):
        cap = None
        consecutive_fails = 0
        open_started_ts = 0.0
        last_frame_ts = 0.0
        last_open_reason = "initial"
        last_open_started_ts = 0.0
        frames_since_open = 0
        try:
            while not self.stop_event.is_set():
                cap_is_open = bool(cap is not None and hasattr(cap, "isOpened") and cap.isOpened())
                if cap is not None and not cap_is_open:
                    now = time.time()
                    last_gap = now - last_frame_ts if last_frame_ts > 0.0 else -1.0
                    open_age = now - open_started_ts if open_started_ts > 0.0 else -1.0
                    next_reconnect = self.reconnect_count + 1
                    self.last_reconnect_reason = "not_opened"
                    self.last_frame_gap = last_gap
                    self.last_open_age = open_age
                    self.frames_since_open = frames_since_open
                    self._log(
                        key=f"wheel.reader.closed.{self.side}",
                        message=(
                            f"[wheel:{self.side}] reader lost opened-state, reconnect #{next_reconnect} "
                            f"reason=not_opened frames_since_open={frames_since_open} "
                            f"last_frame_gap={last_gap:.2f}s open_age={open_age:.2f}s source={self.source}"
                        ),
                        window_seconds=10.0,
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    consecutive_fails = 0
                    self.reconnect_count = next_reconnect
                    last_open_reason = "not_opened"
                    self.last_open_reason = last_open_reason
                    if self.stop_event.wait(self.reconnect_delay):
                        break
                    continue

                if cap is None or not cap_is_open:
                    last_open_started_ts = time.time()
                    cap, decode_meta = create_video_reader(self.source, self.reader_args)
                    if cap is None or not hasattr(cap, "isOpened") or not cap.isOpened():
                        self._log(
                            key=f"wheel.reader.open.{self.side}",
                            message=(
                                f"[wheel:{self.side}] reader open failed, "
                                f"attempt={self.open_count + 1} reopen_count={self.reconnect_count} "
                                f"reason={last_open_reason} source={self.source}"
                            ),
                            window_seconds=10.0,
                        )
                        try:
                            if cap is not None:
                                cap.release()
                        except Exception:
                            pass
                        cap = None
                        if self.stop_event.wait(self.reconnect_delay):
                            break
                        continue
                    mode = str((decode_meta or {}).get("decode_mode") or "sw")
                    backend = str((decode_meta or {}).get("decode_backend") or "software")
                    self.open_count += 1
                    open_started_ts = time.time()
                    frames_since_open = 0
                    open_delay = open_started_ts - last_open_started_ts if last_open_started_ts > 0.0 else 0.0
                    last_gap = open_started_ts - last_frame_ts if last_frame_ts > 0.0 else -1.0
                    self.frames_since_open = 0
                    self.last_open_reason = last_open_reason
                    self.last_open_delay = open_delay
                    self.last_frame_gap = last_gap
                    self.last_open_age = 0.0
                    print(
                        f"[wheel:{self.side}] reader opened mode={mode} backend={backend} "
                        f"open_count={self.open_count} reopen_count={self.reconnect_count} "
                        f"reason={last_open_reason} open_delay={open_delay:.2f}s "
                        f"last_frame_gap={last_gap:.2f}s source={self.source}"
                    )
                    consecutive_fails = 0

                ok, frame = cap.read()
                if not ok or frame is None:
                    failure_summary = self._capture_failure_summary(cap)
                    consecutive_fails += 1
                    if consecutive_fails < self.reader_fail_threshold:
                        if self.stop_event.wait(0.05):
                            break
                        continue
                    now = time.time()
                    next_reconnect = self.reconnect_count + 1
                    last_gap = now - last_frame_ts if last_frame_ts > 0.0 else -1.0
                    open_age = now - open_started_ts if open_started_ts > 0.0 else -1.0
                    self.last_reconnect_reason = "read_fail_threshold"
                    self.last_frame_gap = last_gap
                    self.last_open_age = open_age
                    self.frames_since_open = frames_since_open
                    self._log(
                        key=f"wheel.reader.reconnect.{self.side}",
                        message=(
                            f"[wheel:{self.side}] reader stalled, reconnect #{next_reconnect} "
                            f"reason=read_fail_threshold consecutive_fails={consecutive_fails} "
                            f"threshold={self.reader_fail_threshold} frames_since_open={frames_since_open} "
                            f"last_frame_gap={last_gap:.2f}s open_age={open_age:.2f}s source={self.source}"
                            f"{(' ' + failure_summary) if failure_summary else ''}"
                        ),
                        window_seconds=10.0,
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    consecutive_fails = 0
                    self.reconnect_count = next_reconnect
                    last_open_reason = "read_fail_threshold"
                    self.last_open_reason = last_open_reason
                    if self.stop_event.wait(self.reconnect_delay):
                        break
                    continue

                consecutive_fails = 0
                self.frames += 1
                frames_since_open += 1
                last_frame_ts = time.time()
                self.frames_since_open = frames_since_open
                self.last_frame_ts = last_frame_ts
                self.last_open_age = last_frame_ts - open_started_ts if open_started_ts > 0.0 else -1.0
                self.frame_slot.put((frame, time.time()))
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass


class WheelProcessorThread(threading.Thread):
    def __init__(
        self,
        side,
        model_path,
        class_names,
        frame_slot,
        result_cache,
        stop_event,
        active_event=None,
        boost_event=None,
        target_fps=10.0,
        active_target_fps=None,
        imgsz=640,
        conf_thresh=0.25,
        nms_thresh=0.45,
        core_mask=None,
    ):
        super().__init__(daemon=True)
        self.side = str(side)
        self.model_path = Path(model_path)
        self.class_names = list(class_names or DEFAULT_WHEEL_CLASSES)
        self.frame_slot = frame_slot
        self.result_cache = result_cache
        self.stop_event = stop_event
        self.active_event = active_event
        self.boost_event = boost_event
        self.target_fps = max(float(target_fps or 0.0), 0.1)
        active_fps_raw = self.target_fps if active_target_fps is None else float(active_target_fps or 0.0)
        self.active_target_fps = max(0.0, active_fps_raw)
        self.target_interval = 1.0 / self.target_fps
        self.active_target_interval = 0.0 if self.active_target_fps <= 0.0 else (1.0 / self.active_target_fps)
        self.imgsz = max(64, int(imgsz))
        self.conf_thresh = max(0.0, float(conf_thresh))
        self.nms_thresh = max(0.01, float(nms_thresh))
        self.core_mask = core_mask
        self.frames = 0
        self.infer_time = 0.0
        self.center_hits = 0
        self.last_infer_ts = 0.0
        self.last_infer_duration = 0.0
        self.last_hit_ts = 0.0
        self.last_hit_capture_ts = 0.0
        self._output_mode = "6"
        self._postprocessor = self._build_postprocessor(self._output_mode)
        self._log_throttle = WindowedLogThrottle()

    def _build_postprocessor(self, output_mode):
        return FpModelPostprocessor(
            img_size=(self.imgsz, self.imgsz),
            obj_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            output_mode=output_mode,
            num_classes=len(self.class_names),
        )

    def _ensure_output_mode(self, outputs):
        expected = "9" if len(outputs) == 9 else "6"
        if expected == self._output_mode:
            return
        self._output_mode = expected
        self._postprocessor = self._build_postprocessor(expected)
        print(f"[wheel:{self.side}] fp_postprocess_mode={self._postprocessor.describe_mode()}")

    def _log(self, key, message, window_seconds=10.0):
        self._log_throttle.log(key=key, message=message, window_seconds=window_seconds, emit=print)

    @staticmethod
    def _build_runtime():
        from rknnlite.api import RKNNLite

        return RKNNLite()

    def run(self):
        if not self.model_path.exists():
            print(f"[wheel:{self.side}] model missing: {self.model_path}")
            return

        rk = None
        try:
            rk = self._build_runtime()
            if rk.load_rknn(str(self.model_path)) != 0:
                raise RuntimeError("load_rknn failed")
            init_kwargs = {}
            if self.core_mask is not None:
                init_kwargs["core_mask"] = self.core_mask
            if rk.init_runtime(**init_kwargs) != 0:
                raise RuntimeError("init_runtime failed")
        except Exception as exc:
            print(f"[wheel:{self.side}] runtime init failed: {exc}")
            try:
                if rk is not None:
                    rk.release()
            except Exception:
                pass
            return

        last_seq = 0
        next_infer_ts = 0.0
        try:
            while not self.stop_event.is_set():
                if self.active_event is not None and not self.active_event.is_set():
                    next_infer_ts = 0.0
                    if self.stop_event.wait(0.2):
                        break
                    continue
                boost_active = bool(self.boost_event is not None and self.boost_event.is_set())
                current_interval = self.active_target_interval if boost_active else self.target_interval
                now = time.time()
                if next_infer_ts > now:
                    if self.stop_event.wait(min(next_infer_ts - now, 0.05)):
                        break
                    continue

                seq, item = self.frame_slot.peek()
                if item is None or seq == last_seq:
                    self.frame_slot.wait_for_update(last_seq, timeout=0.2)
                    continue

                last_seq = seq
                frame, capture_ts = item
                started = time.time()
                next_infer_ts = 0.0 if current_interval <= 0.0 else (started + current_interval)
                try:
                    img_input, lb_info = self._postprocessor.prepare(frame)
                    outputs = rk.inference(inputs=[img_input], data_format=["nhwc"])
                    if not outputs:
                        continue
                    self._ensure_output_mode(outputs)
                    boxes, classes, scores = self._postprocessor.postprocess(outputs)
                    if boxes is None or classes is None or scores is None:
                        continue
                    boxes = self._postprocessor.map_boxes_to_original(boxes, lb_info)
                    if self.result_cache.update_from_detections(
                        side=self.side,
                        frame=frame,
                        capture_ts=capture_ts,
                        boxes=boxes,
                        classes=classes,
                        scores=scores,
                        class_names=self.class_names,
                    ):
                        self.center_hits += 1
                        self.last_hit_ts = time.time()
                        self.last_hit_capture_ts = float(capture_ts or 0.0)
                except Exception as exc:
                    self._log(
                        key=f"wheel.processor.{self.side}",
                        message=f"[wheel:{self.side}] inference failed: {exc}",
                        window_seconds=10.0,
                    )
                finally:
                    duration = max(0.0, time.time() - started)
                    self.frames += 1
                    self.infer_time += duration
                    self.last_infer_duration = duration
                    self.last_infer_ts = time.time()
        finally:
            try:
                rk.release()
            except Exception:
                pass


class WheelDetectionService:
    def __init__(self, config, base_dir=None, hw_decode=False, imgsz=640, image_quality=85):
        self.config = config or {}
        self.settings = resolve_wheel_settings(self.config, base_dir=base_dir)
        self.enabled = bool(self.settings.get("enabled"))
        self.event_driven = bool(self.settings.get("event_driven", True))
        self.result_cache = WheelResultCache(
            bind_window_seconds=self.settings["bind_window_seconds"],
            image_quality=image_quality,
        )
        self.stop_event = threading.Event()
        self.inference_active_event = threading.Event()
        self.boost_active_event = threading.Event()
        self.reader_args = SimpleNamespace(
            hw_decode=bool(hw_decode),
            _config=self.config,
        )
        self.reader_fail_threshold = max(1, int((self.config or {}).get("reader_fail_threshold", 5)))
        self.reader_reconnect_delay = max(0.2, float((self.config or {}).get("reader_reconnect_delay", 2.0)))
        configured_imgsz = self.settings.get("imgsz")
        self.imgsz = max(64, int(configured_imgsz if configured_imgsz else imgsz))
        self.core_mask = parse_core_mask(self.settings.get("core_mask"))
        self.streams = {}
        self.active_sides = []
        self._activity_lock = threading.Lock()
        self._active_tracks = {}

    @staticmethod
    def _should_track_activate(track_state):
        if not isinstance(track_state, dict):
            return False
        if track_state.get("closed"):
            return False
        if 5 in (track_state.get("events") or ()):
            return False
        if int(track_state.get("zone_a_enter_frame", -1) or -1) >= 0:
            return True
        return bool(track_state.get("zone_a_dwell_frames", 0) > 0)

    def _refresh_inference_active_locked(self):
        should_run = bool(self._active_tracks) or (not self.event_driven)
        if should_run:
            self.inference_active_event.set()
        else:
            self.inference_active_event.clear()
        if self._active_tracks:
            self.boost_active_event.set()
        else:
            self.boost_active_event.clear()

    def update_track_activity(self, track_id, track_state=None, frame_ts=None, active=None):
        if not self.enabled:
            return
        track_id = int(track_id or 0)
        if track_id <= 0:
            return
        if active is None:
            active = self._should_track_activate(track_state)
        active = bool(active)
        with self._activity_lock:
            if active:
                self._active_tracks[track_id] = time.time() if frame_ts is None else float(frame_ts)
            else:
                self._active_tracks.pop(track_id, None)
            self._refresh_inference_active_locked()

    def forget_track(self, track_id):
        if not self.enabled:
            return
        track_id = int(track_id or 0)
        if track_id <= 0:
            return
        with self._activity_lock:
            self._active_tracks.pop(track_id, None)
            self._refresh_inference_active_locked()

    def start(self):
        if not self.enabled:
            return False

        model_path = Path(self.settings["model_path"])
        if not model_path.exists():
            print(f"[wheel] enabled but model missing: {model_path}")
            return False

        for side in WHEEL_SIDES:
            source = str(self.settings.get(f"{side}_source", "") or "").strip()
            if not source:
                continue
            frame_slot = _LatestFrameSlot()
            reader = WheelReaderThread(
                side=side,
                source=source,
                reader_args=self.reader_args,
                frame_slot=frame_slot,
                stop_event=self.stop_event,
                reader_fail_threshold=self.reader_fail_threshold,
                reconnect_delay=self.reader_reconnect_delay,
            )
            processor = WheelProcessorThread(
                side=side,
                model_path=model_path,
                class_names=self.settings["classes"],
                frame_slot=frame_slot,
                result_cache=self.result_cache,
                stop_event=self.stop_event,
                active_event=self.inference_active_event,
                boost_event=self.boost_active_event,
                target_fps=self.settings["target_fps"],
                active_target_fps=self.settings["active_target_fps"],
                imgsz=self.imgsz,
                conf_thresh=self.settings["conf_thresh"],
                nms_thresh=self.settings["nms_thresh"],
                core_mask=self.core_mask,
            )
            self.streams[side] = {
                "reader": reader,
                "processor": processor,
            }
            reader.start()
            processor.start()
            self.active_sides.append(side)

        if not self.active_sides:
            print("[wheel] enabled but no left/right source configured, skip sidechain")
            return False

        with self._activity_lock:
            self._refresh_inference_active_locked()

        print(
            f"[wheel] enabled sides={','.join(self.active_sides)} "
            f"event_driven={self.event_driven} "
            f"target_fps={self.settings['target_fps']:.2f} "
            f"active_target_fps={self.settings['active_target_fps']:.2f} "
            f"imgsz={self.imgsz} core_mask={self.core_mask}"
        )
        return True

    def stop(self):
        self.stop_event.set()
        for side in list(self.streams.keys()):
            stream = self.streams.get(side) or {}
            for key in ("reader", "processor"):
                worker = stream.get(key)
                if worker is None:
                    continue
                try:
                    worker.join(timeout=1.0)
                except Exception:
                    pass

    def get_recent_results(self, now_ts=None):
        return self.result_cache.get_recent_results(now_ts=now_ts)

    def get_recent_result_entries(self, now_ts=None, reference_ts=None, track_id=None):
        return self.result_cache.get_recent_result_entries(
            now_ts=now_ts,
            reference_ts=reference_ts,
            track_id=track_id,
        )

    def claim_result_entry(self, track_id, entry_id):
        return self.result_cache.claim_result_entry(track_id, entry_id)

    def get_claimed_result_entries(self, track_id, now_ts=None, reference_ts=None):
        return self.result_cache.get_claimed_result_entries(
            track_id=track_id,
            now_ts=now_ts,
            reference_ts=reference_ts,
        )

    def get_photo_candidate_entries(self, track_id, now_ts=None, reference_ts=None):
        return self.result_cache.get_photo_candidate_entries(
            track_id=track_id,
            now_ts=now_ts,
            reference_ts=reference_ts,
        )

    def snapshot_stats(self):
        stats = {}
        with self._activity_lock:
            active_track_ids = sorted(int(track_id) for track_id in self._active_tracks.keys())
            inference_active = bool(self.inference_active_event.is_set())
            boost_active = bool(self.boost_active_event.is_set())
        for side, stream in self.streams.items():
            reader = stream.get("reader")
            processor = stream.get("processor")
            stats[side] = {
                "decode_frames": int(getattr(reader, "frames", 0) or 0),
                "infer_frames": int(getattr(processor, "frames", 0) or 0),
                "infer_time": float(getattr(processor, "infer_time", 0.0) or 0.0),
                "center_hits": int(getattr(processor, "center_hits", 0) or 0),
                "avg_infer_ms": (
                    1000.0 * float(getattr(processor, "infer_time", 0.0) or 0.0)
                    / max(1, int(getattr(processor, "frames", 0) or 0))
                ),
                "last_infer_ms": 1000.0 * float(getattr(processor, "last_infer_duration", 0.0) or 0.0),
                "last_infer_age": (
                    time.time() - float(getattr(processor, "last_infer_ts", 0.0) or 0.0)
                    if float(getattr(processor, "last_infer_ts", 0.0) or 0.0) > 0.0 else -1.0
                ),
                "last_hit_age": (
                    time.time() - float(getattr(processor, "last_hit_ts", 0.0) or 0.0)
                    if float(getattr(processor, "last_hit_ts", 0.0) or 0.0) > 0.0 else -1.0
                ),
                "last_hit_capture_age": (
                    time.time() - float(getattr(processor, "last_hit_capture_ts", 0.0) or 0.0)
                    if float(getattr(processor, "last_hit_capture_ts", 0.0) or 0.0) > 0.0 else -1.0
                ),
                "reader_open_count": int(getattr(reader, "open_count", 0) or 0),
                "reader_reconnect_count": int(getattr(reader, "reconnect_count", 0) or 0),
                "reader_frames_since_open": int(getattr(reader, "frames_since_open", 0) or 0),
                "reader_last_open_reason": str(getattr(reader, "last_open_reason", "") or ""),
                "reader_last_reconnect_reason": str(getattr(reader, "last_reconnect_reason", "") or ""),
                "reader_last_frame_gap": float(getattr(reader, "last_frame_gap", -1.0) or -1.0),
                "reader_last_open_age": float(getattr(reader, "last_open_age", -1.0) or -1.0),
                "reader_last_open_delay": float(getattr(reader, "last_open_delay", 0.0) or 0.0),
                "reader_alive": bool(reader.is_alive()) if reader is not None else False,
                "processor_alive": bool(processor.is_alive()) if processor is not None else False,
            }
        stats["_cache"] = self.result_cache.snapshot_stats()
        stats["_service"] = {
            "event_driven": bool(self.event_driven),
            "inference_active": inference_active,
            "boost_active": boost_active,
            "target_fps": float(self.settings.get("target_fps", 0.0) or 0.0),
            "active_target_fps": float(self.settings.get("active_target_fps", 0.0) or 0.0),
            "active_track_count": len(active_track_ids),
            "active_track_ids": active_track_ids,
        }
        return stats
