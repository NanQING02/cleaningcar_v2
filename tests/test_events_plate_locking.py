import json
import tempfile
import unittest
from collections import deque

from cleaningcar.events import EventManager


class _DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
        return None, {"enter_a": False, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class _ZoneAZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
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

    def update_track(self, track_id, anchor_point, frame_idx, vehicle_height=None):
        spec = self.states.get(frame_idx, {})
        state = type(
            "ZoneState",
            (),
            {
                "inside_a": bool(spec.get("inside_a", False)),
                "inside_b": bool(spec.get("inside_b", False)),
                "born_inside_a": bool(spec.get("born_inside_a", False)),
                "born_inside_pending": bool(spec.get("born_inside_pending", False)),
                "born_inside_outcome": str(spec.get("born_inside_outcome", "") or ""),
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
    def test_initial_plate_lock_always_uses_six_hits(self):
        manager = self._manager(plate_lock_frames=6)
        manager.event_plate_lock_frames = 6

        self.assertEqual(manager._event_plate_lock_required_hits({'speed_buf': [1.0, 3.0]}), 6)
        self.assertEqual(manager._event_plate_lock_required_hits({'speed_buf': [36.0, 42.0]}), 6)

    def test_plate_vehicle_motion_rejects_large_relative_position_jump(self):
        consistent = EventManager._plate_vehicle_motion_consistent(
            [700, 0, 1920, 1080],
            [1400, 900, 1550, 950],
            [680, 0, 1900, 1080],
            [1650, 180, 1780, 230],
        )

        self.assertFalse(consistent)

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

    @staticmethod
    def _update(
        manager,
        frame_idx,
        plate_text="",
        plate_is_guess=False,
        plate_color="",
        plate_color_conf=None,
        plate_conf=0.95,
        plate_text_conf=None,
        plate_mutual_verified=False,
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
            plate_text_conf=plate_text_conf,
            plate_mutual_verified=plate_mutual_verified,
        )

    def test_unlocked_does_not_report_shadow_guess(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=False)
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

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 3, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 4, plate_text="粤B98765", plate_is_guess=False)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 4)

        self.assertEqual(text, "鲁A12345")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_latest"), "粤B98765")
        self.assertEqual(track_state.get("plate_text_locked"), "鲁A12345")

    def test_all_letter_plate_never_enters_lock_or_report(self):
        uploader = _CollectingUploader()
        manager = self._manager(plate_lock_frames=3, uploader=uploader)

        for frame_idx in range(1, 6):
            self._update(manager, frame_idx, plate_text="吉MEJWUN", plate_is_guess=False)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 5)

        self.assertEqual(track_state.get("plate_text_latest"), "")
        self.assertEqual(track_state.get("plate_text_locked"), "")
        self.assertEqual(text, "")
        self.assertFalse(is_guess)

        manager._emit_event_core(
            track_id=1,
            event_type=1,
            frame_idx=5,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:00"},
            track_state=track_state,
            vehicle_type="car",
        )
        manager._emit_event_core(
            track_id=1,
            event_type=2,
            frame_idx=6,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:01"},
            track_state=track_state,
            vehicle_type="car",
        )

        self.assertEqual(len(uploader.payloads), 2)
        for payload in uploader.payloads:
            self.assertEqual(payload["plateNumber"], "")
            self.assertTrue(payload["plateRecognitionAbnormal"])
            self.assertIn("PLATE_NOT_DETECTED", str(payload.get("abnormalReason", "")))

    def test_locked_text_does_not_switch_below_correction_threshold(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=False, plate_conf=0.70)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=False, plate_conf=0.70)
        self._update(manager, 3, plate_text="鲁A12345", plate_is_guess=False, plate_conf=0.70)
        self.assertEqual(manager.tracks[1].get("plate_text_locked"), "鲁A12345")
        manager._emit_event_core(
            track_id=1,
            event_type=2,
            frame_idx=3,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:03"},
            track_state=manager.tracks[1],
            vehicle_type="car",
        )

        self._update(manager, 4, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 5, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 6, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 7, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 8, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 9, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 10, plate_text="粤B98765", plate_is_guess=False, plate_conf=0.99)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 10)
        self.assertEqual(text, "鲁A12345")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_locked"), "鲁A12345")

    def test_weak_valid_candidate_is_retained_but_cannot_lock(self):
        manager = self._manager(plate_lock_frames=3)
        for frame_idx in range(1, 5):
            self._update(manager, frame_idx, plate_text="鲁A12345", plate_conf=0.45)

        track_state = manager.tracks[1]
        shadow_entries = list(manager.shadow_pool.get(1) or ())

        self.assertEqual(track_state.get("plate_text_locked"), "")
        self.assertEqual(len(shadow_entries), 4)
        self.assertFalse(any(entry.get("trusted") for entry in shadow_entries))

    def test_late_wrong_text_below_correction_threshold_keeps_stable_result(self):
        manager = self._manager(plate_lock_frames=3)
        manager.enable_event_disk = True

        for frame_idx in range(1, 6):
            self._update(
                manager,
                frame_idx,
                plate_text="苏A3A329",
                plate_color="黄色",
                plate_color_conf=0.95,
            )
        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_text_locked"), "苏A3A329")
        self.assertEqual(track_state.get("plate_color_locked"), "黄色")

        manager._emit_event_core(
            track_id=1,
            event_type=1,
            frame_idx=5,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:00"},
            track_state=track_state,
            vehicle_type="yellow truck",
        )
        manager._emit_event_core(
            track_id=1,
            event_type=2,
            frame_idx=5,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:01"},
            track_state=track_state,
            vehicle_type="yellow truck",
        )
        for frame_idx in range(6, 14):
            self._update(
                manager,
                frame_idx,
                plate_text="桂N0T7RUK",
                plate_color="蓝色",
                plate_color_conf=0.98,
            )
        self._update(manager, 14, plate_text="", plate_color="黑色", plate_color_conf=0.99)
        track_state = manager.tracks[1]

        manager._emit_event_core(
            track_id=1,
            event_type=5,
            frame_idx=14,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:10"},
            track_state=track_state,
            vehicle_type="yellow truck",
        )

        saved = sorted(manager.events_dir.glob("*_t5_*.json"))
        with saved[0].open("r", encoding="utf-8") as fh:
            event = json.load(fh)
        self.assertEqual(track_state.get("plate_text_locked"), "苏A3A329")
        self.assertEqual(event["plateNumber"], "苏A3A329")
        self.assertEqual(event["plateColor"], "黄色")
        self.assertEqual(track_state.get("plate_color_locked_text"), "苏A3A329")

    def test_twelve_continuous_confirmed_hits_can_correct_after_type2(self):
        manager = self._manager(plate_lock_frames=3)
        manager.enable_event_disk = True

        for frame_idx in range(1, 4):
            self._update(
                manager,
                frame_idx,
                plate_text="苏A3A329",
                plate_color="黄色",
                plate_color_conf=0.95,
            )
        track_state = manager.tracks[1]
        manager._emit_event_core(
            track_id=1,
            event_type=2,
            frame_idx=3,
            frame=None,
            payload={"captureTime": "2026-07-18 10:00:01"},
            track_state=track_state,
            vehicle_type="yellow truck",
        )

        for frame_idx in range(4, 16):
            self._update(
                manager,
                frame_idx,
                plate_text="桂N0T7RUK",
                plate_color="蓝色",
                plate_color_conf=0.98,
            )

        self.assertEqual(track_state.get("plate_text_locked"), "桂N0T7RUK")
        self.assertEqual(track_state.get("plate_color_locked"), "蓝色")
        self.assertEqual(track_state.get("plate_color_locked_text"), "桂N0T7RUK")

    def test_correction_hits_must_be_continuous(self):
        manager = self._manager(plate_lock_frames=3)
        for frame_idx in range(1, 4):
            self._update(manager, frame_idx, plate_text="苏A3A329")

        for frame_idx in range(4, 10):
            self._update(manager, frame_idx, plate_text="桂N0T7RUK")
        self._update(manager, 10, plate_text="苏A3A329")
        for frame_idx in range(11, 22):
            self._update(manager, frame_idx, plate_text="桂N0T7RUK")

        self.assertEqual(manager.tracks[1].get("plate_text_locked"), "苏A3A329")
        self._update(manager, 22, plate_text="桂N0T7RUK")
        self.assertEqual(manager.tracks[1].get("plate_text_locked"), "桂N0T7RUK")

    def test_white_and_black_colors_are_ignored_by_business_layer(self):
        manager = self._manager(plate_lock_frames=3)
        for frame_idx, color in enumerate(("白色", "黑色", "白色", "黑色"), start=1):
            self._update(
                manager,
                frame_idx,
                plate_text="苏A3A329",
                plate_color=color,
                plate_color_conf=0.99,
            )

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_text_locked"), "苏A3A329")
        self.assertEqual(track_state.get("plate_color_locked"), "")
        self.assertEqual(track_state.get("plate_color_evidence_by_text"), {})

    def test_color_requires_trusted_text_evidence(self):
        manager = self._manager(plate_lock_frames=3)
        for frame_idx in range(1, 5):
            self._update(
                manager,
                frame_idx,
                plate_text="苏A3A329",
                plate_text_conf=0.40,
                plate_color="黄色",
                plate_color_conf=0.99,
            )

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_text_locked"), "")
        self.assertEqual(track_state.get("plate_color_evidence_by_text"), {})

    def test_plate_only_empty_scene_cannot_replace_stable_text_or_color(self):
        manager = self._manager(plate_lock_frames=3)
        for frame_idx in range(1, 4):
            self._update(
                manager,
                frame_idx,
                plate_text="苏A3A329",
                plate_color="黄色",
                plate_color_conf=0.95,
            )

        for frame_idx in range(4, 20):
            manager.update_track(
                track_id=1,
                plate_box=[0, 0, 10, 10],
                vehicle_box=None,
                plate_text="桂N0T7RUK",
                frame_idx=frame_idx,
                frame=None,
                water_boxes=[],
                water_active=False,
                is_plate=True,
                vehicle_label="",
                vehicle_conf=None,
                plate_conf=0.99,
                confirmed=False,
                plate_color="蓝色",
                plate_color_conf=0.99,
                plate_type="single",
            )

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_text_locked"), "苏A3A329")
        self.assertEqual(track_state.get("plate_color_locked"), "黄色")
        self.assertNotIn("桂N0T7RUK", track_state.get("plate_color_evidence_by_text", {}))

    def test_yellow_truck_fallback_is_used_without_bound_color_evidence(self):
        manager = self._manager(plate_lock_frames=3)
        for frame_idx in range(1, 4):
            self._update(manager, frame_idx, plate_text="苏A3A329")
        manager.tracks[1]['vehicle_cls'] = 'yellow truck'

        color, confidence = manager._infer_plate_color(manager.tracks[1])

        self.assertEqual(color, "黄色")
        self.assertEqual(confidence, 0.0)
        self.assertEqual(manager.tracks[1]['plate_color_source'], 'vehicle_type_fallback')

    def test_unlocked_event_marks_plate_recognition_abnormal_and_blank_plate(self):
        manager = self._manager(plate_lock_frames=3)
        manager.enable_event_disk = True

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=False)
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

        self._update(manager, 1, plate_text="鲁A12345", plate_is_guess=False)
        self._update(manager, 2, plate_text="鲁A12345", plate_is_guess=False)
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

    def test_build_api_payload_type6_includes_per_id_video_enabled(self):
        uploader = _CollectingUploader()
        manager = self._manager(plate_lock_frames=3, uploader=uploader)

        payload = manager._build_api_payload(
            {
                "id": "session-1",
                "trackId": 1,
                "type": 6,
                "perIdVideoEnabled": False,
            },
            {},
            frame_idx=2,
        )

        self.assertEqual(payload["id"], "session-1")
        self.assertEqual(payload["type"], 6)
        self.assertEqual(payload["lane"], "lane-a")
        self.assertFalse(payload["perIdVideoEnabled"])

    def test_record_tail_frames_only_when_per_id_video_enabled(self):
        manager = self._manager(plate_lock_frames=3)

        manager.per_id_video_enabled = False
        self.assertEqual(manager._record_tail_frames(), 0)

        manager.per_id_video_enabled = True
        self.assertEqual(manager._record_tail_frames(), 200)

    def test_color_lock_on_majority_high_confidence(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.95)
        self._update(manager, 2, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.92)
        self._update(manager, 3, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.91)

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_color_locked"), "蓝色")
        self.assertIn("蓝色", track_state.get("plate_color_votes", {}))
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "蓝色")
        self.assertGreater(confidence, 0.9)

    def test_locked_color_wins_over_low_confidence_new_color(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.95)
        self._update(manager, 2, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.92)
        self._update(manager, 3, plate_text="鲁A12345", plate_color="blue", plate_color_conf=0.91)
        self._update(manager, 4, plate_text="鲁A12345", plate_color="yellow", plate_color_conf=0.10)
        self._update(manager, 5, plate_text="鲁A12345", plate_color="yellow", plate_color_conf=0.20)

        color, confidence = manager._infer_plate_color(manager.tracks[1])

        self.assertEqual(color, "蓝色")
        self.assertGreater(confidence, 0.9)

    def test_color_falls_back_to_vehicle_heuristic_only_without_model_signal(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="鲁A1234D")

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_color_locked"), "")
        self.assertEqual(track_state.get("plate_color_latest"), "")
        color, confidence = manager._infer_plate_color(track_state)
        self.assertTrue(color)
        self.assertEqual(confidence, 0.0)
        self.assertEqual(track_state['plate_color_source'], 'vehicle_type_fallback')

        track_state["plate_color_latest"] = "green"
        track_state["plate_color_latest_conf"] = 0.66
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "绿色")
        self.assertAlmostEqual(confidence, 0.66, places=6)

        track_state["plate_color_locked"] = "blue"
        track_state["plate_color_locked_conf"] = 0.88
        track_state["plate_color_locked_text"] = track_state.get("plate_text_locked", "")
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "蓝色")
        self.assertAlmostEqual(confidence, 0.88, places=6)
        self.assertTrue(color)
        self.assertGreater(confidence, 0.0)

    def test_type1_uses_five_stable_zone_a_frames_without_confidence_gate(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 6)
        })
        manager = self._manager_with_zone(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        for idx in range(1, 5):
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

        manager.update_track(
            1, None, [0, 0, 20, 20], "", 5, None, [], False, False,
            "car", 0.40, None, False, anchor_point=(10.0, 10.0),
        )

        self.assertIn(1, emitted)
        self.assertIn(1, manager.tracks[1]["events"])

    def test_type1_does_not_depend_on_motion_direction(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 7)
        })
        manager = self._manager_with_zone(zone)
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

    def test_plate_candidate_does_not_bypass_type1_stable_frame_gate(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 3)
        })
        manager = self._manager_with_zone(zone)
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

        self.assertNotIn(1, emitted)
        self.assertNotIn(1, manager.tracks[1]["events"])

    def test_type3_requires_three_consecutive_water_frames_after_type2(self):
        zone = _ScriptedZoneManager({
            1: {"inside_a": True, "enter_a": True},
            2: {"inside_a": True, "inside_b": True, "enter_b": True},
            3: {"inside_a": True, "inside_b": True},
            4: {"inside_a": True, "inside_b": True},
            5: {"inside_a": True, "inside_b": True},
        })
        manager = self._manager_with_zone(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        for frame_idx in range(1, 6):
            water_boxes = [[0, 0, 5, 5]] if frame_idx >= 3 else []
            manager.update_track(
                1, None, [0, 0, 20, 20], "", frame_idx, None,
                water_boxes, bool(water_boxes), False, "car", 0.95, None, False,
                anchor_point=(10.0, 10.0),
            )
            if frame_idx == 3:
                manager.update_track(
                    1, None, [0, 0, 20, 20], "", frame_idx, None,
                    water_boxes, True, False, "car", 0.95, None, False,
                    anchor_point=(10.0, 10.0),
                )
            if frame_idx < 5:
                self.assertNotIn(3, emitted)

        self.assertEqual(emitted, [1, 2, 3])
        self.assertEqual(manager.tracks[1]["water_consecutive_frames"], 3)

    def test_type4_uses_half_second_zone_b_dwell_at_source_fps(self):
        manager = self._manager()

        self.assertEqual(manager.min_type4_zone_b_dwell_seconds, 0.5)
        self.assertEqual(manager.min_type4_zone_b_dwell, 13)

    def test_type1_requests_temporary_video_rotation_without_closing_lifecycle(self):
        zone = _ScriptedZoneManager({
            idx: {"inside_a": True, "enter_a": idx == 1}
            for idx in range(1, 5)
        })
        manager = self._manager_with_zone(zone)
        manager.min_type1_track_frames = 1
        manager.per_id_video_enabled = True
        manager.pre_type2_video_segment_seconds = 0.08

        self._update(manager, 1)
        self._update(manager, 2)
        self.assertFalse(manager.tracks[1]["pre_type2_rotate_requested"])
        self._update(manager, 3)

        track_state = manager.tracks[1]
        self.assertEqual(track_state["events"], {1})
        self.assertTrue(track_state["pre_type2_rotate_requested"])
        self.assertFalse(track_state["type2_qualified"])
        self.assertFalse(track_state.get("closed", False))
        self.assertEqual(track_state["record_first_start_frame"], 1)
        self.assertEqual(track_state["record_segment_start_frame"], 1)

    def test_type2_commits_video_and_cancels_pending_temporary_rotation(self):
        zone = _ScriptedZoneManager({
            1: {"inside_a": True, "enter_a": True},
            2: {"inside_a": True, "inside_b": True, "enter_b": True},
        })
        manager = self._manager_with_zone(zone)
        manager.min_type1_track_frames = 1

        self._update(manager, 1)
        manager.tracks[1]["pre_type2_rotate_requested"] = True
        self._update(manager, 2)

        track_state = manager.tracks[1]
        self.assertEqual(track_state["events"], {1, 2})
        self.assertTrue(track_state["recording_committed"])
        self.assertFalse(track_state["pre_type2_rotate_requested"])

    def test_type2_dwell_timeout_forces_type4_type5_and_immediate_video_stop(self):
        zone = _ScriptedZoneManager({
            1: {"inside_a": True, "enter_a": True},
            2: {"inside_a": True, "inside_b": True, "enter_b": True},
            3: {"inside_a": True, "inside_b": True},
            4: {"inside_a": True, "inside_b": True},
        })
        manager = self._manager_with_zone(zone)
        manager.min_type1_track_frames = 1
        manager.post_type2_force_finalize_seconds = 0.08
        manager.enable_event_disk = True

        for frame_idx in range(1, 5):
            self._update(manager, frame_idx)

        track_state = manager.tracks[1]
        self.assertEqual(track_state["events"], {1, 2, 4, 5})
        self.assertTrue(track_state["closed"])
        self.assertEqual(track_state["record_stop_frame"], 4)
        self.assertIn("OVER_15_MINUTES_AFTER_TYPE2", track_state["abnormal_reasons"])
        type5_path = manager.events_dir / f"{manager.camera_id}_1_t5_4.json"
        payload = json.loads(type5_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["forcedCompletionReason"], "type2_dwell_timeout_15m")
        self.assertIn("OVER_15_MINUTES_AFTER_TYPE2", payload["abnormalReason"])

    def test_vehicle_class_lock_weights_type2_to_type4_votes_without_short_override(self):
        manager = self._manager()
        track_state = {
            'class_counts': {'yellow truck': 1013, 'dump truck': 33},
            'wash_stage_class_counts': {'yellow truck': 894, 'dump truck': 33},
            'vehicle_cls': 'yellow truck',
            'vehicle_cls_locked': '',
            'vehicle_cls_frozen': False,
            'vehicle_cls_at_type2': 'yellow truck',
            'last_vehicle_label': 'dump truck',
        }

        selected = manager._finalize_vehicle_class(track_state, 'type4')

        self.assertEqual(selected, 'yellow truck')
        self.assertEqual(track_state['vehicle_cls_locked'], 'yellow truck')
        self.assertTrue(track_state['vehicle_cls_frozen'])
        self.assertEqual(track_state['vehicle_cls_lock_reason'], 'type4')

    def test_vehicle_class_lock_can_follow_sustained_wash_stage_evidence(self):
        manager = self._manager()
        track_state = {
            'class_counts': {'yellow truck': 10, 'dump truck': 9},
            'wash_stage_class_counts': {'dump truck': 5},
            'vehicle_cls': 'yellow truck',
            'vehicle_cls_locked': '',
            'vehicle_cls_frozen': False,
            'vehicle_cls_at_type2': 'yellow truck',
            'last_vehicle_label': 'dump truck',
        }

        selected = manager._finalize_vehicle_class(track_state, 'type5')

        self.assertEqual(selected, 'dump truck')
        self.assertEqual(track_state['vehicle_cls_locked'], 'dump truck')
        self.assertEqual(track_state['vehicle_cls_lock_reason'], 'type5')

    def test_zone_a_reentry_clears_historical_exit_marker(self):
        zone = _ScriptedZoneManager({
            1: {"inside_a": True, "enter_a": True},
            2: {"inside_a": False, "exit_a": True},
            3: {"inside_a": True, "enter_a": True},
        })
        manager = self._manager_with_zone(zone)

        self._update(manager, 1)
        self._update(manager, 2)
        self.assertTrue(manager.tracks[1]["zone_a_exited"])

        self._update(manager, 3)

        self.assertFalse(manager.tracks[1]["zone_a_exited"])

    def test_type5_waits_for_anchor_disappearance_after_zone_exit(self):
        zone = _ScriptedZoneManager({
            1: {"inside_a": True, "enter_a": True},
            2: {"inside_a": True, "inside_b": True, "enter_b": True},
            3: {"inside_a": True, "inside_b": True},
            4: {"inside_a": True, "inside_b": True},
            5: {"inside_a": False, "inside_b": True, "exit_a": True},
            6: {"inside_a": False, "inside_b": True},
            7: {"inside_a": False, "inside_b": False, "exit_b": True},
        })
        manager = self._manager_with_zone(zone)
        manager.min_type4_zone_b_dwell = 0
        emitted = []

        def fake_emit(track_id, event_type, frame_idx, frame, payload, track_state):
            del frame, payload
            if manager._event_stage_allowed(track_id, event_type, frame_idx, track_state):
                emitted.append(event_type)
                return True
            return False

        manager.emit_event = fake_emit
        for frame_idx in range(1, 6):
            self._update(manager, frame_idx)

        track_state = manager.tracks[1]
        self.assertEqual(emitted, [1, 2])
        self.assertFalse(track_state.get("closed", False))

        self._update(manager, 6)
        self.assertEqual(emitted, [1, 2])
        self.assertFalse(track_state.get("closed", False))

        self._update(manager, 7)
        self.assertEqual(emitted, [1, 2, 4])
        self.assertFalse(track_state.get("closed", False))

        manager.flush_inactive(
            active_ids=set(),
            frame_idx=7 + manager.timeout_frames,
        )

        self.assertEqual(emitted, [1, 2, 4, 5])
        self.assertTrue(track_state["closed"])
        self.assertEqual(track_state["event_sequence_issues"], [])

    def test_zone_a_reentry_does_not_close_active_track(self):
        zone = _ScriptedZoneManager({
            1: {"inside_a": True, "enter_a": True},
            2: {"inside_a": True, "inside_b": True, "enter_b": True},
            3: {"inside_a": False, "inside_b": True, "exit_a": True},
            4: {"inside_a": True, "inside_b": True, "enter_a": True},
            5: {"inside_a": True, "inside_b": False, "exit_b": True},
        })
        manager = self._manager_with_zone(zone)
        manager.min_type4_zone_b_dwell = 0
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        for frame_idx in range(1, 6):
            self._update(manager, frame_idx)

        track_state = manager.tracks[1]
        self.assertEqual(emitted, [1, 2, 4])
        self.assertNotIn("type5_pending_exit_a", track_state)
        self.assertFalse(track_state.get("closed", False))

    def test_born_inside_suppresses_type1_until_zone_b_promotes_candidate(self):
        zone = _ScriptedZoneManager({
            1: {
                "inside_a": True,
                "enter_a": True,
                "born_inside_a": True,
                "born_inside_pending": True,
                "born_inside_outcome": "CANDIDATE",
            },
            2: {
                "inside_a": True,
                "inside_b": True,
                "enter_b": True,
                "born_inside_a": True,
                "born_inside_outcome": "PROMOTED",
            },
        })
        manager = self._manager_with_zone(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        self._update(manager, 1)
        self.assertEqual(emitted, [])
        self.assertTrue(manager.tracks[1]["born_inside_pending"])

        self._update(manager, 2)
        self.assertEqual(emitted, [1, 2])
        self.assertFalse(manager.tracks[1]["born_inside_pending"])
        self.assertEqual(manager.tracks[1]["born_inside_outcome"], "PROMOTED")

    def test_born_inside_pass_by_never_emits_type1(self):
        zone = _ScriptedZoneManager({
            1: {
                "inside_a": True,
                "enter_a": True,
                "born_inside_a": True,
                "born_inside_pending": True,
                "born_inside_outcome": "CANDIDATE",
            },
            2: {
                "inside_a": False,
                "exit_a": True,
                "born_inside_a": True,
                "born_inside_outcome": "PASS_BY",
            },
        })
        manager = self._manager_with_zone(zone)
        emitted = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)

        self._update(manager, 1)
        self._update(manager, 2)

        self.assertEqual(emitted, [])
        self.assertEqual(manager.tracks[1]["born_inside_outcome"], "PASS_BY")

    def test_born_inside_candidate_timeout_becomes_rejected_without_events(self):
        manager = self._manager()
        emitted = []
        traces = []
        manager.emit_event = lambda track_id, event_type, *args, **kwargs: emitted.append(event_type)
        manager.trace_record = lambda kind, data: traces.append((kind, data))
        manager.tracks[1] = {
            "events": set(),
            "last_frame_idx": 0,
            "last_frame": None,
            "zone_a_dwell_frames": 20,
            "born_inside_a": True,
            "born_inside_pending": True,
            "born_inside_outcome": "CANDIDATE",
            "type2_qualified": False,
            "vehicle_hit_frames": 20,
            "last_vehicle_box": [0, 0, 20, 20],
        }

        manager.flush_inactive(active_ids=set(), frame_idx=manager.timeout_frames + 1)

        self.assertEqual(emitted, [])
        self.assertNotIn(1, manager.tracks)
        self.assertEqual(traces[-1][0], "born_inside")
        self.assertEqual(traces[-1][1]["action"], "rejected")
        self.assertEqual(traces[-1][1]["reason"], "track_lost_in_zone_a_timeout")

    def test_lost_formal_lifecycle_marks_type4_and_type5_abnormal(self):
        manager = self._manager()
        emitted = []

        def fake_emit(track_id, event_type, frame_idx, frame, payload, track_state):
            del track_id, frame_idx, frame
            emitted.append((event_type, set(track_state["abnormal_reasons"]), dict(payload)))

        manager.emit_event = fake_emit
        manager.tracks[1] = {
            "events": {1, 2},
            "last_frame_idx": 0,
            "last_frame": None,
            "zone_a_seen": True,
            "zone_a_exited": False,
            "zone_a_dwell_frames": 20,
            "zone_b_dwell_frames": 20,
            "type2_qualified": True,
            "vehicle_hit_frames": 20,
            "last_vehicle_box": [0, 0, 20, 20],
            "abnormal_reasons": set(),
        }

        manager.flush_inactive(active_ids=set(), frame_idx=manager.timeout_frames + 1)

        self.assertEqual([event_type for event_type, _, _ in emitted], [4, 5])
        self.assertTrue(emitted[0][2]["sequenceBackfill"])
        self.assertEqual(emitted[0][2]["backfillReason"], "type5_anchor_missing_timeout")
        for _, reasons, _ in emitted:
            self.assertIn("TRACK_LOST_IN_ZONE_A_TIMEOUT", reasons)
            self.assertNotIn("MISSING_TYPE4", reasons)

    def test_file_eof_does_not_mark_formal_lifecycle_as_track_loss(self):
        manager = self._manager()
        emitted = []

        def fake_emit(track_id, event_type, frame_idx, frame, payload, track_state):
            del track_id, frame_idx, frame, payload
            emitted.append((event_type, set(track_state["abnormal_reasons"])))

        manager.emit_event = fake_emit
        manager.tracks[1] = {
            "events": {1, 2},
            "last_frame_idx": 0,
            "last_frame": None,
            "zone_a_seen": True,
            "zone_a_exited": False,
            "zone_a_dwell_frames": 20,
            "zone_b_dwell_frames": 20,
            "type2_qualified": True,
            "vehicle_hit_frames": 20,
            "last_vehicle_box": [0, 0, 20, 20],
            "abnormal_reasons": set(),
        }

        manager.flush_inactive(
            active_ids=set(),
            frame_idx=manager.timeout_frames + 1,
            timeout_reason="file_eof",
        )

        self.assertEqual([event_type for event_type, _ in emitted], [4, 5])
        for _, reasons in emitted:
            self.assertNotIn("TRACK_LOST_IN_ZONE_A_TIMEOUT", reasons)
            self.assertNotIn("MISSING_TYPE4", reasons)

    def test_lost_lifecycle_uses_locked_anchor_direction_when_zone_exit_unknown(self):
        manager = self._manager()
        track_state = {
            "zone_state": None,
            "abnormal_reasons": {"TRACK_LOST_IN_ZONE_A_TIMEOUT"},
            "anchor_motion_direction": "forward",
            "anchor_direction_confidence": 0.86,
            "anchor_direction_locked": True,
        }

        direction_code, direction_label = manager._resolve_direction(track_state)

        self.assertEqual((direction_code, direction_label), (5, "正向前出"))
        self.assertEqual(track_state["direction_source"], "trajectory_inference")

    def test_lost_lifecycle_keeps_unknown_direction_without_locked_anchor(self):
        manager = self._manager()
        track_state = {
            "zone_state": None,
            "abnormal_reasons": {"TRACK_LOST_IN_ZONE_A_TIMEOUT"},
            "anchor_motion_direction": "forward",
            "anchor_direction_confidence": 0.86,
            "anchor_direction_locked": False,
        }

        self.assertEqual(manager._resolve_direction(track_state), (0, ""))
        self.assertEqual(track_state["direction_source"], "unknown")

    def test_pending_plate_history_stays_untrusted_after_vehicle_binding(self):
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

        self.assertEqual(manager.tracks[101]["plate_text_locked"], "")
        self.assertEqual(manager.tracks[101]["plate_text"], "鲁A12345")

    def test_type5_requires_type2_but_not_water_signal(self):
        manager = self._manager()
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

    def test_type5_is_not_suppressed_by_low_confidence_or_track_jump(self):
        manager = self._manager()
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

    def test_type5_still_emits_when_vehicle_type_is_missing(self):
        manager = self._manager()
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

        self.assertEqual([entry[0][1] for entry in emitted], [4, 5])
        self.assertIn(4, track_state["events"])
        self.assertIn(5, track_state["events"])

    def test_flush_inactive_always_closes_type2_qualified_lifecycle(self):
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

        self.assertEqual(emitted, [4, 5])
        self.assertEqual(track_state["events"], {1, 2, 4, 5})

    def test_flush_inactive_does_not_wait_tail_when_per_id_video_disabled(self):
        manager = self._manager()
        manager.per_id_video_enabled = False
        finalized = []
        track_state = {
            "events": {1, 2, 5},
            "last_frame_idx": 0,
            "last_frame": None,
            "record_start_frame": 0,
            "record_stop_frame": None,
            "type2_qualified": True,
            "zone_a_dwell_frames": 20,
            "vehicle_hit_frames": 20,
            "last_vehicle_box": [0, 0, 20, 20],
            "track_frame_count": 20,
        }
        manager.tracks[7] = track_state

        manager.flush_inactive(
            active_ids=set(),
            frame_idx=manager.timeout_frames + 1,
            on_track_timeout=lambda track_id, state: finalized.append(track_id),
        )

        self.assertEqual(finalized, [7])
        self.assertEqual(track_state["record_stop_frame"], 0)
        self.assertNotIn(7, manager.tracks)

    def test_track_timeout_type5_does_not_append_per_id_tail(self):
        manager = self._manager()
        manager.per_id_video_enabled = True
        manager._emit_event_core = lambda *args, **kwargs: None
        track_state = {
            "events": {1, 2},
            "last_frame_idx": 40,
            "last_frame": None,
            "type2_qualified": True,
            "zone_a_seen": True,
            "zone_a_exited": False,
            "zone_a_dwell_frames": 20,
            "vehicle_hit_frames": 20,
            "last_vehicle_box": [0, 0, 20, 20],
            "track_frame_count": 20,
            "record_start_frame": 0,
            "record_stop_frame": None,
        }
        manager.tracks[8] = track_state

        manager.flush_inactive(active_ids=set(), frame_idx=40 + manager.timeout_frames + 1)

        self.assertIn(5, track_state["events"])
        self.assertEqual(track_state["record_stop_frame"], manager.timeout_frames + 41)


if __name__ == "__main__":
    unittest.main()
