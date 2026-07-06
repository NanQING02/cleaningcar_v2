from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.parse
import urllib.request

from .log_throttle import WindowedLogThrottler


def _camera_namespace(camera_id):
    raw = str(camera_id or "default").encode("utf-8", errors="ignore")
    digest = hashlib.blake2s(raw, digest_size=2).digest()
    return max(1, int.from_bytes(digest, "big"))


def _decode_entry(item):
    if not isinstance(item, dict):
        return item
    decoded = dict(item)
    encoded = decoded.pop("imageJpegBase64", "")
    if encoded and not decoded.get("imageJpegBytes"):
        try:
            decoded["imageJpegBytes"] = base64.b64decode(str(encoded).encode("ascii"))
        except Exception:
            decoded["imageJpegBytes"] = b""
    return decoded


class RemoteWheelResultProvider:
    def __init__(self, base_url, camera_id="default", timeout=0.5):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout = max(0.1, float(timeout or 0.5))
        self.namespace = _camera_namespace(camera_id)
        self.active_sides = ("left", "right")
        self._log_throttler = WindowedLogThrottler()

    def _record_error(self, key, message):
        for line in self._log_throttler.record(
            key=key,
            message=f"[wheel-remote] {message}",
            now=time.time(),
            window_seconds=10.0,
        ):
            print(line)

    def _remote_track_id(self, track_id):
        try:
            local_id = int(track_id or 0)
        except (TypeError, ValueError):
            local_id = 0
        if local_id <= 0:
            return 0
        return self.namespace * 1000000 + local_id

    def _request_json(self, method, path, payload=None, query=None):
        if not self.base_url:
            raise RuntimeError("wheel remote service url is empty")
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def update_track_activity(self, track_id, track_state=None, frame_ts=None, active=None):
        remote_id = self._remote_track_id(track_id)
        if remote_id <= 0:
            return
        try:
            self._request_json(
                "POST",
                "/activity",
                {
                    "track_id": remote_id,
                    "active": bool(active),
                    "frame_ts": time.time() if frame_ts is None else float(frame_ts),
                },
            )
        except Exception as exc:
            self._record_error("activity", f"activity update failed: {exc}")

    def forget_track(self, track_id):
        self.update_track_activity(track_id, active=False)

    def get_recent_result_entries(self, now_ts=None, reference_ts=None, track_id=None):
        remote_id = self._remote_track_id(track_id)
        try:
            payload = self._request_json(
                "GET",
                "/recent",
                query={
                    "track_id": remote_id,
                    "now_ts": time.time() if now_ts is None else float(now_ts),
                    "reference_ts": time.time() if reference_ts is None else float(reference_ts),
                },
            )
        except Exception as exc:
            self._record_error("recent", f"recent results request failed: {exc}")
            return []
        return [_decode_entry(item) for item in (payload or {}).get("items", [])]

    def claim_result_entry(self, track_id, entry_id):
        remote_id = self._remote_track_id(track_id)
        try:
            payload = self._request_json(
                "POST",
                "/claim",
                {"track_id": remote_id, "entry_id": int(entry_id or 0)},
            )
        except Exception as exc:
            self._record_error("claim", f"claim request failed: {exc}")
            return False
        return bool((payload or {}).get("ok"))

    def get_photo_candidate_entries(self, track_id, now_ts=None, reference_ts=None):
        remote_id = self._remote_track_id(track_id)
        try:
            payload = self._request_json(
                "GET",
                "/photo-candidates",
                query={
                    "track_id": remote_id,
                    "now_ts": time.time() if now_ts is None else float(now_ts),
                    "reference_ts": time.time() if reference_ts is None else float(reference_ts),
                },
            )
        except Exception as exc:
            self._record_error("photo-candidates", f"photo candidates request failed: {exc}")
            return []
        return [_decode_entry(item) for item in (payload or {}).get("items", [])]

    def get_claimed_result_entries(self, track_id, now_ts=None, reference_ts=None):
        remote_id = self._remote_track_id(track_id)
        try:
            payload = self._request_json(
                "GET",
                "/claimed",
                query={
                    "track_id": remote_id,
                    "now_ts": time.time() if now_ts is None else float(now_ts),
                    "reference_ts": time.time() if reference_ts is None else float(reference_ts),
                },
            )
        except Exception as exc:
            self._record_error("claimed", f"claimed results request failed: {exc}")
            return []
        return [_decode_entry(item) for item in (payload or {}).get("items", [])]

    def snapshot_stats(self):
        try:
            payload = self._request_json("GET", "/stats")
        except Exception as exc:
            self._record_error("stats", f"stats request failed: {exc}")
            return {}
        sides = (payload or {}).get("active_sides")
        if isinstance(sides, list):
            self.active_sides = tuple(str(side) for side in sides)
        return (payload or {}).get("stats", {})
