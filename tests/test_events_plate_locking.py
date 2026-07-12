import json
import tempfile
import unittest
from collections import deque

from cleaningcar.events import EventManager


class _DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx):
        return None, {"enter_a": False, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class _ZoneAZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx):
        state = type("ZoneState", (), {"inside_a": True, "inside_b": False})()
        return state, {"enter_a": frame_idx == 1, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class _ScriptedZoneManager:
    def __init__(self, states):
        self.states = states

    def update_track(self, track_id, anchor_point, frame_idx):
        spec = self.states.get(frame_idx, {})
        state = type(
            "ZoneState",
            (),
            {
                "inside_a": bool(spec.get("inside_a", False)),
                "inside_b": bool(spec.get("inside_b", False)),
            },
        )()
        flags = {
            "enter_a": bool(spec.get("enter_a", False)),
            "exit_a": bool(spec.get("exit_a", False)),
            "enter_b": bool(spec.get("enter_b", False)),
            "exit_b": bool(spec.get("exit_b", False)),
        }
        return state, flags

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class _WheelActivityProvider:
    def __init__(self):
        self.calls = []

    def update_track_activity(self, track_id, track_state=None, frame_ts=None, active=None):
        self.calls.append(
            {
                "track_id": int(track_id),
                "active": bool(active),
                "zone_a_dwell_frames": int((track_state or {}).get("zone_a_dwell_frames", 0) or 0),
                "closed": bool((track_state or {}).get("closed")),
            }
        )


class _CollectingUploader:
    def __init__(self):
        self.payloads = []

    def enqueue(self, payload):
        self.payloads.append(dict(payload))


class EventManagerPlateLockingTests(unittest.TestCase):
    def _manager(self, plate_lock_frames=3, uploader=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config = {
            "logic": {
                "plate_lock_frames": plate_lock_frames,
                "default_plate_color": "",
                "default_plate_color_conf": 0.0,
            },
            "shadow_pool": {
                "max_candidates": 80,
                "max_age_frames": 120,
                "text_window_frames": 20,
                "text_margin_ratio": 0.10,
                "text_switch_min_consecutive": 3,
                "text_switch_gain_ratio": 1.20,
                "text_switch_margin_ratio": 0.18,
                "color_min_confidence": 0.60,
                "color_lock_frames": 3,
                "color_window_frames": 20,
                "color_switch_min_consecutive": 3,
                "color_switch_gain_ratio": 1.20,
                "color_switch_margin": 0.5,
            },
            "event_capture_dir": temp_dir.name,
            "event_output_dir": temp_dir.name,
            "lane_name": "lane-a",
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_DummyZoneManager(),
            uploader=uploader,
        )

    def _manager_with_zone(self, zone_manager, wheel_provider=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config = {
            "logic": {
                "plate_lock_frames": 3,
                "default_plate_color": "",
                "default_plate_color_conf": 0.0,
            },
            "shadow_pool": {
                "max_candidates": 80,
                "max_age_frames": 120,
            },
            "event_capture_dir": temp_dir.name,
            "event_output_dir": temp_dir.name,
            "lane_name": "lane-a",
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=zone_manager,
            wheel_result_provider=wheel_provider,
        )

    def _quality_manager(self, zone_manager):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config = {
            "logic": {
                "plate_lock_frames": 3,
                "event_track_quality": {
                    "enabled": True,
                    "min_hits_type1": 12,
                    "fast_vehicle_min_hits_type1": 6,
                    "min_avg_vehicle_conf": 0.62,
                    "fast_vehicle_min_avg_conf": 0.72,
                    "plate_candidate_min_hits": 2,
                    "plate_candidate_can_confirm_type1": True,
                    "min_zone_a_dwell_type5": 15,
                    "suppress_obvious_false_type5": True,
                    "suspicious_cooldown_seconds": 6,
                },
            },
            "shadow_pool": {
                "max_candidates": 80,
                "max_age_frames": 120,
                "text_window_frames": 20,
                "text_switch_min_consecutive": 3,
                "color_min_confidence": 0.60,
                "color_lock_frames": 3,
            },
            "event_capture_dir": temp_dir.name,
            "event_output_dir": temp_dir.name,
            "lane_name": "lane-a",
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=zone_manager,
        )

    @staticmethod
    def _update(
        manager,
        frame_idx,
        plate_text="",
        plate_is_guess=False,
        plate_color="",
        plate_color_conf=None,
        plate_conf=0.95,
    ):
        manager.update_track(
            1,
            [0, 0, 10, 10],
            [0, 0, 20, 20],
            plate_text,
            frame_idx,
            None,
            [],
            False,
            True,
            "car",
            0.95,
            plate_conf,
            False,
            anchor_point=(5.0, 5.0),
            plate_is_guess=plate_is_guess,
            plate_color=plate_color,
            plate_color_conf=plate_color_conf,
            plate_type="single",
        )

    def test_unlocked_does_not_report_shadow_guess(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 3, plate_text="粤B98765", plate_is_guess=True)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 3)

        self.assertEqual(text, "")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_latest"), "粤B98765")
        self.assertEqual(track_state.get("plate_text_locked"), "")

    def test_locked_text_is_not_overwritten_by_single_wrong_frame(self):
        manager = self._manager(plate_lock_frames=3)
        manager.enable_event_disk = True

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 3, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 4, plate_text="粤B98765", plate_is_guess=False)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 4)

        self.assertEqual(text, "鲁A12345")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_latest"), "粤B98765")
        self.assertEqual(track_state.get("plate_text_locked"), "鲁A12345")

    def test_text_can_switch_after_stronger_consecutive_candidate(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=False, plate_conf=0.60)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=False, plate_conf=0.60)
        self._update(manager, 3, plate_text="鲁A12345", plate_is_guess=False, plate_conf=0.60)
        self.assertEqual(manager.tracks[1].get("plate_text_locked"), "鲁A12345")

        self._update(manager, 4, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 5, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 6, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 7, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 8, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 9, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 10, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 10)
        self.assertEqual(text, "粤B98765")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_locked"), "粤B98765")

    def test_unlocked_event_marks_plate_recognition_abnormal_and_blank_plate(self):
        manager = self._manager(plate_lock_frames=3)
        manager.enable_event_disk = True

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=True)
        track_state = manager.tracks[1]

        manager._emit_event_core(
            track_id=1,
            event_type=1,
            frame_idx=2,
            frame=None,
            payload={"captureTime": "2026-07-04 20:00:00"},
            track_state=track_state,
            vehicle_type="car",
        )

        saved = sorted(manager.events_dir.glob("*_t1_*.json"))
        self.assertEqual(len(saved), 1)
        with saved[0].open("r", encoding="utf-8") as fh:
            event = json.load(fh)
        self.assertEqual(event["plateNumber"], "")
        self.assertFalse(event["plateIsGuess"])
        self.assertTrue(event["plateRecognitionAbnormal"])
        self.assertIn("PLATE_NOT_LOCKED", str(event.get("abnormalReason", "")))

    def test_buffered_type1_payload_uses_locked_plate_when_type2_flushes(self):
        uploader = _CollectingUploader()
        manager = self._manager(plate_lock_frames=3, uploader=uploader)

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=True)
        track_state = manager.tracks[1]
        manager._emit_event_core(
            track_id=1,
            event_type=1,
            frame_idx=2,
            frame=None,
            payload={"captureTime": "2026-07-04 20:01:00"},
            track_state=track_state,
            vehicle_type="car",
        )
        self.assertEqual(len(uploader.payloads), 0)

        self._update(manager, 3, plate_text="鲁A12345", plate_is_guess=False)
        track_state = manager.tracks[1]
        manager._emit_event_core(
            track_id=1,
            event_type=2,
            frame_idx=3,
            frame=None,
            payload={"captureTime": "2026-07-04 20:01:01"},
            track_state=track_state,
            vehicle_type="car",
        )

        self.assertEqual(len(uploader.payloads), 2)
        first_payload = uploader.payloads[0]
        self.assertEqual(first_payload["type"], 1)
        self.assertEqual(first_payload["plateNumber"], "鲁A12345")
        self.assertFalse(first_payload["plateRecognitionAbnormal"])
        self.assertFalse(first_payload["plateIsGuess"])

    def test_build_api_payload_marks_plate_not_locked_abnormal_reason(self):
        uploader = _CollectingUploader()
        manager = self._manager(plate_lock_frames=3, uploader=uploader)

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=True)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=True)
        track_state = manager.tracks[1]

        payload = manager._build_api_payload(
            {
                "id": "session-1",
                "trackId": 1,
                "type": 3,
                "captureTime": "2026-07-04 20:02:00",
                "captureImage": "",
                "plateNumber": "",
                "vehicleType": "car",
                "plateRecognitionAbnormal": True,
            },
            track_state,
            frame_idx=2,
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["plateNumber"], "")
        self.assertEqual(payload["plateConfidence"], 0.0)
        self.assertTrue(payload["plateRecognitionAbnormal"])
        self.assertTrue(payload["isAbnormal"])
        self.assertIn("PLATE_NOT_LOCKED", str(payload.get("abnormalReason", "")))

    def test_color_lock_on_majority_high_confidence(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.95)
        self._update(manager, 2, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.92)
        self._update(manager, 3, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.91)

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_color_locked"), "blue")
        self.assertIn("blue", track_state.get("plate_color_votes", {}))
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "blue")
        self.assertGreater(confidence, 0.9)

    def test_locked_color_wins_over_low_confidence_new_color(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.95)
        self._update(manager, 2, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.92)
        self._update(manager, 3, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.91)
        self._update(manager, 4, plate_text="鲁A12345", plate_color="yellow", plate_color_conf=0.10)
        self._update(manager, 5, plate_text="鲁A12345", plate_color="yellow", plate_color_conf=0.20)

        color, confidence = manager._infer_plate_color(manager.tracks[1])

        self.assertEqual(color, "blue")
        self.assertGreater(confidence, 0.9)

    def test_color_falls_back_to_vehicle_heuristic_only_without_model_signal(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A1234D")

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_color_locked"), "")
        self.assertEqual(track_state.get("plate_color_latest"), "")
        color, confidence = manager._infer_plate_color(track_state)
        self.assertTrue(color)
        self.assertGreater(confidence, 0.0)

        track_state["plate_color_latest"] = "green"
        track_state["plate_color_latest_conf"] = 0.66
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "green")
        self.assertAlmostEqual(confidence, 0.66, places=6)

        track_state["plate_color_locked"] = "blue"
        track_state["plate_color_locked_conf"] = 0.88
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "blue")
        self.assertAlmostEqual(confidence, 0.88, places=6)
        self.assertTrue(color)
        self.assertGreater(confidence, 0.0)

    def test_quality_gate_suppresses_short_low_confidence_type1_without_plate(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 6)
        })
        manager = self._quality_manager(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        for idx in range(1, 6):
            manager.update_track(
                1,
                None,
                [0, 0, 20, 20],
                "",
                idx,
                None,
                [],
                False,
                False,
                "car",
                0.40,
                None,
                False,
                anchor_point=(10.0, 10.0),
            )

        self.assertNotIn(1, emitted)
        self.assertNotIn(1, manager.tracks[1]["events"])

    def test_fast_high_confidence_stable_motion_can_emit_type1(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 7)
        })
        manager = self._quality_manager(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        for idx in range(1, 7):
            manager.update_track(
                1,
                None,
                [idx * 8, 0, idx * 8 + 20, 20],
                "",
                idx,
                None,
                [],
                False,
                False,
                "car",
                0.95,
                None,
                False,
                anchor_point=(float(idx * 8), 10.0),
            )

        self.assertIn(1, emitted)
        self.assertIn(1, manager.tracks[1]["events"])

    def test_valid_plate_candidate_can_confirm_type1_with_few_vehicle_frames(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 3)
        })
        manager = self._quality_manager(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        for idx in range(1, 3):
            manager.update_track(
                1,
                [0, 0, 10, 10],
                [0, 0, 20, 20],
                "鲁A12345",
                idx,
                None,
                [],
                False,
                True,
                "car",
                0.55,
                0.95,
                False,
                anchor_point=(5.0 + idx, 5.0),
                plate_is_guess=False,
            )

        self.assertIn(1, emitted)
        self.assertIn(1, manager.tracks[1]["events"])

    def test_pending_plate_history_can_lock_after_vehicle_binding(self):
        manager = self._manager(plate_lock_frames=3)

        manager.update_track(
            101,
            None,
            [0, 0, 40, 40],
            "",
            20,
            None,
            [],
            False,
            False,
            "car",
            0.92,
            None,
            True,
            anchor_point=(20.0, 20.0),
            plate_candidate_history=[
                {"text": "鲁A12345", "conf": 0.91, "frame": 10, "trusted": True},
                {"text": "鲁A12345", "conf": 0.93, "frame": 11, "trusted": True},
                {"text": "鲁A12345", "conf": 0.94, "frame": 12, "trusted": True},
            ],
        )

        self.assertEqual(manager.tracks[101]["plate_text_locked"], "鲁A12345")
        self.assertEqual(manager.tracks[101]["plate_text"], "鲁A12345")

    def test_type5_requires_type2_but_not_water_signal(self):
        manager = self._quality_manager(_DummyZoneManager())
        track_state = {
            "events": {1, 2},
            "type2_qualified": True,
            "zone_a_seen": True,
            "zone_a_dwell_frames": 6,
            "vehicle_conf_history": [0.95] * 6,
            "vehicle_hit_frames": 6,
            "track_frame_count": 6,
        }

        self.assertTrue(manager._can_emit_type5(track_state))

    def test_obvious_false_type5_is_suppressed_and_recorded_in_cooldown(self):
        manager = self._quality_manager(_DummyZoneManager())
        track_state = {
            "events": {1, 2},
            "type2_qualified": True,
            "zone_a_seen": True,
            "zone_a_dwell_frames": 14,
            "vehicle_conf_history": [0.40] * 3,
            "vehicle_hit_frames": 3,
            "track_frame_count": 3,
            "last_vehicle_box": [0, 0, 20, 20],
            "center_jump_history": deque([0.20]),
        }

        self.assertTrue(manager._can_emit_type5(track_state))
        self.assertTrue(manager._should_suppress_type5_quality(1, track_state, 20))
        self.assertTrue(manager.suspicious_type5_cooldown)

    def test_update_track_notifies_wheel_activity_when_track_enters_zone_a(self):
        provider = _WheelActivityProvider()
        manager = self._manager_with_zone(_ZoneAZoneManager(), wheel_provider=provider)

        self._update(manager, 1, plate_text="鲁A12345")

        self.assertTrue(provider.calls)
        self.assertTrue(provider.calls[-1]["active"])
        self.assertGreaterEqual(provider.calls[-1]["zone_a_dwell_frames"], 0)

    def test_flush_inactive_notifies_wheel_activity_false(self):
        provider = _WheelActivityProvider()
        manager = self._manager_with_zone(_ZoneAZoneManager(), wheel_provider=provider)

        self._update(manager, 1, plate_text="鲁A12345")
        provider.calls.clear()
        manager.flush_inactive(active_ids=set(), frame_idx=100)

        self.assertTrue(provider.calls)
        self.assertFalse(provider.calls[-1]["active"])

    def test_type5_still_emits_when_vehicle_type_required_but_missing(self):
        manager = self._manager()
        manager.require_vehicle_type_for_events = True
        emitted = []

        def fake_emit(*args, **kwargs):
            emitted.append((args, kwargs))

        manager._emit_event_core = fake_emit
        track_state = {
            "events": {1, 2},
            "last_frame_idx": 0,
            "last_frame": None,
            "type2_qualified": True,
            "zone_a_dwell_frames": 1,
            "vehicle_hit_frames": 1,
            "last_vehicle_box": [0, 0, 20, 20],
        }
        manager.tracks[1] = track_state

        manager.flush_inactive(active_ids=set(), frame_idx=manager.timeout_frames + 1)

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0][1], 5)
        self.assertIn(5, track_state["events"])
        self.assertNotIn(1, manager.pending_events)

    def test_flush_inactive_does_not_emit_plate_only_events(self):
        manager = self._manager()
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)
        track_state = {
            "events": {1, 2},
            "last_frame_idx": 0,
            "last_frame": None,
            "type2_qualified": True,
            "zone_a_dwell_frames": 20,
            "zone_b_dwell_frames": 20,
            "vehicle_hit_frames": 0,
            "last_vehicle_box": None,
            "track_frame_count": 20,
            "plate_candidate_hits": 3,
        }
        manager.tracks[11] = track_state

        manager.flush_inactive(active_ids=set(), frame_idx=manager.timeout_frames + 1)

        self.assertEqual(emitted, [])
        self.assertEqual(track_state["events"], {1, 2})


if __name__ == "__main__":
    unittest.main()
