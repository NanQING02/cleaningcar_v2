import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from config_manager import ConfigManager
from cleaningcar.wheel import (
    WheelDetectionService,
    WheelReaderThread,
    _LatestFrameSlot,
    _safe_source_label,
    create_wheel_video_reader,
    resolve_wheel_settings,
)
from cleaningcar.wheel_gstreamer import (
    admitted_rtp_sequences,
    is_target_video_rtp_caps,
    redact_rtsp_credentials,
    sanitize_wheel_rtp_caps,
)
from web.config_tiers import CONFIG_FIELD_REGISTRY, TIER_DEVELOPER


VIDEO_CAPS = {
    "media": "video",
    "payload": 96,
    "clock-rate": 90000,
    "encoding-name": "H265",
    "seqnum-base": 1,
    "clock-base": 0,
    "ssrc": 1234,
    "sprop-vps": "vps",
    "sprop-sps": "sps",
    "sprop-pps": "pps",
}


class _FakeCapture:
    def __init__(self, opened=True, frame=None, diagnostics=None):
        self.opened = opened
        self.frame = frame
        self.released = False
        self._diagnostics = dict(diagnostics or {})

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.frame is None:
            return False, None
        return True, self.frame.copy()

    def release(self):
        self.released = True

    def get(self, prop_id):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 20.0,
        }
        return values.get(prop_id, 0.0)

    def diagnostics(self):
        return dict(self._diagnostics)


class WheelRtpInfoCapsTests(unittest.TestCase):
    def test_caps_sanitizer_removes_only_broken_bases(self):
        cleaned = sanitize_wheel_rtp_caps(VIDEO_CAPS)

        self.assertNotIn("seqnum-base", cleaned)
        self.assertNotIn("clock-base", cleaned)
        for key, value in VIDEO_CAPS.items():
            if key not in {"seqnum-base", "clock-base"}:
                self.assertEqual(cleaned[key], value)
        self.assertEqual(VIDEO_CAPS["seqnum-base"], 1)
        self.assertEqual(VIDEO_CAPS["clock-base"], 0)

    def test_compatibility_disabled_preserves_original_caps(self):
        self.assertEqual(sanitize_wheel_rtp_caps(VIDEO_CAPS, enabled=False), VIDEO_CAPS)

    def test_only_target_video_track_is_sanitized(self):
        audio = dict(VIDEO_CAPS, media="audio", encoding_name="PCMA")
        audio["encoding-name"] = "PCMA"
        metadata = dict(VIDEO_CAPS, media="application")

        self.assertTrue(is_target_video_rtp_caps(VIDEO_CAPS))
        self.assertFalse(is_target_video_rtp_caps(audio))
        self.assertFalse(is_target_video_rtp_caps(metadata))
        self.assertEqual(sanitize_wheel_rtp_caps(audio), audio)
        self.assertEqual(sanitize_wheel_rtp_caps(metadata), metadata)

    def test_compliant_server_still_admits_first_packet_after_sanitizing(self):
        actual = [5102, 5103, 5104]
        compliant_caps = dict(VIDEO_CAPS, **{"seqnum-base": actual[0]})

        self.assertEqual(admitted_rtp_sequences(actual, compliant_caps["seqnum-base"]), actual)
        self.assertEqual(admitted_rtp_sequences(actual, None), actual)

    def test_high_half_regression_no_longer_waits_for_wrap(self):
        actual = list(range(52713, 65536)) + [0, 1, 2]

        original = admitted_rtp_sequences(actual, seqnum_base=1)
        fixed = admitted_rtp_sequences(actual, seqnum_base=None)

        self.assertEqual(original, [1, 2])
        self.assertEqual(fixed[0], 52713)
        self.assertEqual(fixed[-5:], [65534, 65535, 0, 1, 2])
        self.assertEqual(len(fixed), len(actual))

    def test_low_half_and_16_bit_wrap_remain_continuous(self):
        low = [1, 2, 3, 32767, 32768]
        boundary = [65534, 65535, 0, 1, 2]

        self.assertEqual(admitted_rtp_sequences(low, None), low)
        self.assertEqual(admitted_rtp_sequences(boundary, None), boundary)
        self.assertEqual(len(set(boundary)), len(boundary))


class WheelReaderIntegrationTests(unittest.TestCase):
    def _args(self, enabled=True):
        return SimpleNamespace(
            hw_decode=True,
            _config={
                "video": {
                    "rtsp_latency_ms": 180,
                    "rtsp_appsink_max_buffers": 1,
                    "reader_frame_timeout_seconds": 5.0,
                },
                "wheel": {"ignore_broken_rtp_info": enabled},
            },
        )

    def test_wheel_factory_uses_dedicated_capture_and_keeps_media_metadata(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture = _FakeCapture(opened=True, frame=frame)
        states = []

        with patch("cleaningcar.wheel.create_wheel_gstreamer_capture", return_value=capture) as create_capture:
            source = "rtsp://" + "test-user" + ":" + "test-pass" + "@wheel.invalid/stream1"
            cap, meta = create_wheel_video_reader(
                source,
                self._args(enabled=True),
                side="left",
                reconnect_count=3,
                state_callback=states.append,
            )

        self.assertIs(cap, capture)
        self.assertEqual(meta["decode_backend"], "wheel_gstreamer")
        self.assertEqual(meta["source_kind"], "rtsp")
        kwargs = create_capture.call_args.kwargs
        self.assertEqual(kwargs["side"], "left")
        self.assertEqual(kwargs["latency_ms"], 180)
        self.assertEqual(kwargs["reconnect_count"], 3)
        self.assertTrue(kwargs["ignore_broken_rtp_info"])
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1920.0)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1080.0)
        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 20.0)

    def test_switch_off_returns_to_original_reader(self):
        original = _FakeCapture(opened=True)
        with patch("cleaningcar.wheel.create_video_reader", return_value=(original, {"decode_backend": "gstreamer"})) as create:
            cap, meta = create_wheel_video_reader(
                "rtsp://wheel/stream1",
                self._args(enabled=False),
                side="right",
            )

        self.assertIs(cap, original)
        self.assertEqual(meta["decode_backend"], "gstreamer")
        create.assert_called_once()

    def test_one_failed_side_does_not_block_other_side(self):
        stop_event = threading.Event()
        frame = np.ones((16, 16, 3), dtype=np.uint8)
        right_cap = _FakeCapture(opened=True, frame=frame)

        def fake_factory(source, _args, side, reconnect_count=0, state_callback=None, cancel_event=None):
            del source, reconnect_count, cancel_event
            if side == "left":
                if state_callback:
                    state_callback("failed")
                return _FakeCapture(opened=False, diagnostics={"last_read_error": "no_data"}), {}
            if state_callback:
                state_callback("bgr_ready")
            return right_cap, {"decode_mode": "hw", "decode_backend": "wheel_gstreamer"}

        left_slot = _LatestFrameSlot()
        right_slot = _LatestFrameSlot()
        left = WheelReaderThread("left", "rtsp://left", self._args(), left_slot, stop_event, reconnect_delay=0.2)
        right = WheelReaderThread("right", "rtsp://right", self._args(), right_slot, stop_event, reconnect_delay=0.2)
        with patch("cleaningcar.wheel.create_wheel_video_reader", side_effect=fake_factory):
            left.start()
            right.start()
            deadline = time.time() + 1.0
            while time.time() < deadline and right.frames < 1:
                time.sleep(0.01)
            stop_event.set()
            left.join(timeout=1.0)
            right.join(timeout=1.0)

        self.assertGreaterEqual(right.frames, 1)
        self.assertGreaterEqual(right_slot.peek()[0], 1)
        self.assertGreaterEqual(left.open_attempt_count, 1)
        self.assertEqual(left.reader_state, "failed")

    def test_failed_initial_opens_increment_attempt_and_reconnect_counts(self):
        stop_event = threading.Event()
        frame_slot = _LatestFrameSlot()
        frame = np.ones((16, 16, 3), dtype=np.uint8)
        captures = [
            _FakeCapture(opened=False, diagnostics={"last_read_error": "open_failed_1"}),
            _FakeCapture(opened=False, diagnostics={"last_read_error": "open_failed_2"}),
            _FakeCapture(opened=True, frame=frame),
        ]

        def fake_factory(*_args, state_callback=None, **_kwargs):
            capture = captures.pop(0)
            if state_callback:
                state_callback("bgr_ready" if capture.isOpened() else "failed")
            if not captures:
                stop_event.set()
            return capture, {"decode_mode": "hw", "decode_backend": "fake"}

        reader = WheelReaderThread(
            "left",
            "rtsp://wheel-left",
            self._args(),
            frame_slot,
            stop_event,
            reconnect_delay=0.2,
        )
        with patch("cleaningcar.wheel.create_wheel_video_reader", side_effect=fake_factory):
            reader.start()
            reader.join(timeout=2.0)

        self.assertFalse(reader.is_alive())
        self.assertEqual(reader.open_attempt_count, 3)
        self.assertEqual(reader.reconnect_count, 2)
        self.assertEqual(reader.open_count, 1)
        self.assertEqual(reader.last_open_reason, "open_failed")

    def test_source_label_never_contains_credentials(self):
        username = "test-user"
        credential = "test-pass"
        source = "rtsp://" + username + ":" + credential + "@" + "192.0.2.10:554/stream1?token=query-value"
        label = _safe_source_label(source)
        self.assertEqual(label, "rtsp://192.0.2.10:554/stream1")
        self.assertNotIn(username, label)
        self.assertNotIn(credential, label)
        error = redact_rtsp_credentials(f"failed location={source} user={username} credential={credential}", source)
        self.assertNotIn(username, error)
        self.assertNotIn(credential, error)

    def test_service_snapshot_exposes_reader_readiness_and_diagnostics(self):
        reader = SimpleNamespace(
            frames=10,
            open_count=1,
            open_attempt_count=2,
            reconnect_count=1,
            frames_since_open=10,
            reader_state="bgr_ready",
            last_open_reason="read_fail_threshold",
            last_reconnect_reason="read_fail_threshold",
            last_frame_gap=0.0,
            last_open_age=2.0,
            last_open_delay=0.8,
            last_stale_seconds=0.0,
            is_alive=lambda: True,
            capture_diagnostics=lambda: {"first_rtp_seqnum": 52713, "caps_sanitized": True},
        )
        processor = SimpleNamespace(
            frames=0,
            infer_time=0.0,
            center_hits=0,
            last_infer_duration=0.0,
            last_infer_ts=0.0,
            last_hit_ts=0.0,
            last_hit_capture_ts=0.0,
            is_alive=lambda: True,
        )
        service = WheelDetectionService({"wheel": {"enabled": False}})
        service.streams = {"left": {"reader": reader, "processor": processor}}

        status = service.snapshot_stats()["left"]

        self.assertEqual(status["reader_state"], "bgr_ready")
        self.assertTrue(status["reader_ready"])
        self.assertEqual(status["reader_open_attempt_count"], 2)
        self.assertEqual(status["reader_diagnostics"]["first_rtp_seqnum"], 52713)

    def test_config_default_and_developer_visibility(self):
        settings = resolve_wheel_settings({"wheel": {}})
        entry = next(item for item in CONFIG_FIELD_REGISTRY if item["path"] == "wheel.ignore_broken_rtp_info")

        self.assertTrue(settings["ignore_broken_rtp_info"])
        self.assertEqual(entry["tier"], TIER_DEVELOPER)

    def test_config_manager_normalizes_compatibility_switch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "system": {"device_id": "wheel-test"},
                        "video": {"source": "demo.mp4"},
                        "wheel": {"ignore_broken_rtp_info": "false"},
                        "zones": {
                            "zone_a_detection": [[0, 0], [1, 0], [1, 1]],
                            "zone_b_wash": [[0, 0], [1, 0], [1, 1]],
                            "flow_vector": {"start": [0, 0], "end": [1, 1]},
                        },
                    }
                ),
                encoding="utf-8",
            )

            manager = ConfigManager(path)

        self.assertFalse(manager.data["wheel"]["ignore_broken_rtp_info"])


if __name__ == "__main__":
    unittest.main()
