import tempfile
import unittest

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


class EventManagerPlateLockingTests(unittest.TestCase):
    def _manager(self, plate_lock_frames=3):
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
        return EventManager(config, fps=25.0, frame_size=(128, 128), zone_manager=_DummyZoneManager())

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

    def test_unlocked_uses_shadow_best_guess_instead_of_latest(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="ABC1234", plate_is_guess=True)
        self._update(manager, 2, plate_text="ABC1234", plate_is_guess=True)
        self._update(manager, 3, plate_text="XYZ9999", plate_is_guess=False)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 3)

        self.assertEqual(text, "ABC1234")
        self.assertTrue(is_guess)
        self.assertEqual(track_state.get("plate_text_latest"), "XYZ9999")
        self.assertEqual(track_state.get("plate_text_locked"), "")

    def test_locked_text_is_not_overwritten_by_single_wrong_frame(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="ABC1234", plate_is_guess=True)
        self._update(manager, 2, plate_text="ABC1234", plate_is_guess=True)
        self._update(manager, 3, plate_text="ABC1234", plate_is_guess=False)
        self._update(manager, 4, plate_text="XYZ9999", plate_is_guess=False)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 4)

        self.assertEqual(text, "ABC1234")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_latest"), "XYZ9999")
        self.assertEqual(track_state.get("plate_text_locked"), "ABC1234")

    def test_text_can_switch_after_stronger_consecutive_candidate(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="ABC1234", plate_is_guess=False, plate_conf=0.60)
        self._update(manager, 2, plate_text="ABC1234", plate_is_guess=False, plate_conf=0.60)
        self._update(manager, 3, plate_text="ABC1234", plate_is_guess=False, plate_conf=0.60)
        self.assertEqual(manager.tracks[1].get("plate_text_locked"), "ABC1234")

        self._update(manager, 4, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 5, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 6, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 7, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 8, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 9, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)
        self._update(manager, 10, plate_text="XYZ9999", plate_is_guess=False, plate_conf=0.99)

        track_state = manager.tracks[1]
        text, is_guess = manager._resolve_plate_with_shadow(1, track_state, 10)
        self.assertEqual(text, "XYZ9999")
        self.assertFalse(is_guess)
        self.assertEqual(track_state.get("plate_text_locked"), "XYZ9999")

    def test_color_lock_on_majority_high_confidence(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="ABC1234", plate_color="blue", plate_color_conf=0.95)
        self._update(manager, 2, plate_text="ABC1234", plate_color="blue", plate_color_conf=0.92)
        self._update(manager, 3, plate_text="ABC1234", plate_color="blue", plate_color_conf=0.91)

        track_state = manager.tracks[1]
        self.assertEqual(track_state.get("plate_color_locked"), "blue")
        self.assertIn("blue", track_state.get("plate_color_votes", {}))
        color, confidence = manager._infer_plate_color(track_state)
        self.assertEqual(color, "blue")
        self.assertGreater(confidence, 0.9)

    def test_locked_color_wins_over_low_confidence_new_color(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="ABC1234", plate_color="blue", plate_color_conf=0.95)
        self._update(manager, 2, plate_text="ABC1234", plate_color="blue", plate_color_conf=0.92)
        self._update(manager, 3, plate_text="ABC1234", plate_color="blue", plate_color_conf=0.91)
        self._update(manager, 4, plate_text="ABC1234", plate_color="yellow", plate_color_conf=0.10)
        self._update(manager, 5, plate_text="ABC1234", plate_color="yellow", plate_color_conf=0.20)

        color, confidence = manager._infer_plate_color(manager.tracks[1])

        self.assertEqual(color, "blue")
        self.assertGreater(confidence, 0.9)

    def test_color_falls_back_to_vehicle_heuristic_only_without_model_signal(self):
        manager = self._manager(plate_lock_frames=3)

        self._update(manager, 1, plate_text="ABC12345")

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

    def test_update_track_notifies_wheel_activity_when_track_enters_zone_a(self):
        provider = _WheelActivityProvider()
        manager = self._manager_with_zone(_ZoneAZoneManager(), wheel_provider=provider)

        self._update(manager, 1, plate_text="ABC1234")

        self.assertTrue(provider.calls)
        self.assertTrue(provider.calls[-1]["active"])
        self.assertGreaterEqual(provider.calls[-1]["zone_a_dwell_frames"], 0)

    def test_flush_inactive_notifies_wheel_activity_false(self):
        provider = _WheelActivityProvider()
        manager = self._manager_with_zone(_ZoneAZoneManager(), wheel_provider=provider)

        self._update(manager, 1, plate_text="ABC1234")
        provider.calls.clear()
        manager.flush_inactive(active_ids=set(), frame_idx=100)

        self.assertTrue(provider.calls)
        self.assertFalse(provider.calls[-1]["active"])


if __name__ == "__main__":
    unittest.main()
