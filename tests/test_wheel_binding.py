import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cleaningcar.events import EventManager
from cleaningcar.runtime_config import load_config
from cleaningcar.wheel import WheelResultCache, resolve_wheel_class_name, resolve_wheel_settings


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
        state = SimpleNamespace(inside_a=True, inside_b=False)
        return state, {"enter_a": frame_idx == 1, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class _StaticWheelProvider:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.claims = {}
        self.entry_cluster_seconds = 1.5

    def get_recent_result_entries(self, now_ts=None, reference_ts=None, track_id=None):
        track_id = int(track_id or 0)
        results = []
        for item in self.items:
            copied = dict(item)
            entry_id = int(copied.get("entryId", 0) or 0)
            owner = int(self.claims.get(entry_id, 0) or 0)
            if owner > 0 and owner != track_id:
                continue
            results.append(copied)
        return results

    def claim_result_entry(self, track_id, entry_id):
        track_id = int(track_id or 0)
        entry_id = int(entry_id or 0)
        if track_id <= 0 or entry_id <= 0:
            return False
        target = None
        for item in self.items:
            if int(item.get("entryId", 0) or 0) == entry_id:
                target = item
                break
        if not target:
            return False
        owner = int(self.claims.get(entry_id, 0) or 0)
        if owner > 0 and owner != track_id:
            return False
        target_ts = float(target.get("capture_ts", 0.0) or 0.0)
        target_side = str(target.get("side") or "").strip().lower()
        for item in self.items:
            item_id = int(item.get("entryId", 0) or 0)
            if item_id <= 0:
                continue
            if str(item.get("side") or "").strip().lower() != target_side:
                continue
            item_ts = float(item.get("capture_ts", 0.0) or 0.0)
            if abs(item_ts - target_ts) > self.entry_cluster_seconds:
                continue
            item_owner = int(self.claims.get(item_id, 0) or 0)
            if item_owner > 0 and item_owner != track_id:
                continue
            self.claims[item_id] = track_id
        return True


class WheelBindingTests(unittest.TestCase):
    def _manager(self, wheel_provider=None, zone_manager=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config = {
            "logic": {
                "default_plate_color": "",
                "default_plate_color_conf": 0.0,
            },
            "wheel": {
                "bind_pre_start_seconds": 3.0,
                "bind_after_end_seconds": 3.0,
                "bind_require_active": True,
                "bind_wait_seconds": 0.0,
            },
            "shadow_pool": {
                "max_candidates": 20,
                "max_age_frames": 30,
            },
            "event_capture_dir": temp_dir.name,
            "event_output_dir": temp_dir.name,
            "lane_name": "lane-a",
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=zone_manager or _DummyZoneManager(),
            uploader=object(),
            wheel_result_provider=wheel_provider,
            wheel_photo_base_dir=temp_dir.name,
        )

    @staticmethod
    def _type5_event(capture_time="2026-04-28 12:00:00"):
        return {
            "id": "evt-1",
            "trackId": 1,
            "type": 5,
            "captureTime": capture_time,
            "captureImage": "",
            "plateNumber": "鲁A12345",
            "videoDuration": 12.5,
        }

    @staticmethod
    def _track_state():
        return {
            "plate_conf_history": [0.98],
            "vehicle_conf_history": [0.91],
            "wash_end_time": "2026-04-28 12:00:00",
            "last_frame_idx": 30,
            "wash_duration": 7.3,
            "zone_a_enter_frame": 0,
            "zone_a_dwell_frames": 1,
            "wheel_active": True,
        }

    def test_class_id_maps_to_expected_bucket_names(self):
        class_names = ["0-25", "25-50", "50-75", "75-100"]

        self.assertEqual(resolve_wheel_class_name(class_names, 0), "0-25")
        self.assertEqual(resolve_wheel_class_name(class_names, 1), "25-50")
        self.assertEqual(resolve_wheel_class_name(class_names, 2), "50-75")
        self.assertEqual(resolve_wheel_class_name(class_names, 3), "75-100")

    def test_any_detection_updates_cache_regardless_of_position(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        classes = np.array([3], dtype=np.int64)
        scores = np.array([0.93], dtype=np.float32)

        updated = cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1000.0,
            boxes=np.array([[0.0, 40.0, 18.0, 60.0]], dtype=np.float32),
            classes=classes,
            scores=scores,
            class_names=["0-25", "25-50", "50-75", "75-100"],
        )
        self.assertTrue(updated)

        results = cache.get_recent_results(now_ts=1000.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["side"], "left")
        self.assertEqual(results[0]["className"], "75-100")
        self.assertTrue(results[0]["imageBase64"])

    def test_window_prefers_highest_score(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((100, 100, 3), 150, dtype=np.uint8)
        class_names = ["0-25", "25-50", "50-75", "75-100"]

        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1000.0,
            boxes=np.array([[24.0, 20.0, 84.0, 80.0]], dtype=np.float32),
            classes=np.array([0], dtype=np.int64),
            scores=np.array([0.95], dtype=np.float32),
            class_names=class_names,
        )
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1001.0,
            boxes=np.array([[35.0, 35.0, 65.0, 65.0]], dtype=np.float32),
            classes=np.array([1], dtype=np.int64),
            scores=np.array([0.60], dtype=np.float32),
            class_names=class_names,
        )
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1002.0,
            boxes=np.array([[35.0, 35.0, 65.0, 65.0]], dtype=np.float32),
            classes=np.array([2], dtype=np.int64),
            scores=np.array([0.88], dtype=np.float32),
            class_names=class_names,
        )

        results = cache.get_recent_results(now_ts=1002.0, reference_ts=1002.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["className"], "0-25")

    def test_type5_payload_only_attaches_simplified_wheel_fields(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((80, 80, 3), 120, dtype=np.uint8)
        now_ts = time.time()
        capture_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=now_ts,
            boxes=np.array([[28.0, 28.0, 52.0, 52.0]], dtype=np.float32),
            classes=np.array([2], dtype=np.int64),
            scores=np.array([0.88], dtype=np.float32),
            class_names=["0-25", "25-50", "50-75", "75-100"],
        )

        manager = self._manager(wheel_provider=cache)
        track_state = self._track_state()
        manager._update_track_wheel_results(1, track_state, frame_ts=now_ts)
        payload = manager._build_api_payload(self._type5_event(capture_time=capture_time), track_state, frame_idx=30)

        self.assertIn("wheelResults", payload)
        self.assertEqual(len(payload["wheelResults"]), 1)
        self.assertEqual(
            set(payload["wheelResults"][0].keys()),
            {"side", "captureTime", "photoUrl", "className"},
        )
        self.assertEqual(payload["wheelResults"][0]["side"], "left")
        self.assertEqual(payload["wheelResults"][0]["className"], "50-75")
        self.assertTrue(Path(payload["wheelResults"][0]["photoUrl"]).is_absolute())
        self.assertTrue(Path(payload["wheelResults"][0]["photoUrl"]).exists())

    def test_type5_payload_does_not_fallback_to_provider_without_lifecycle_lock(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((80, 80, 3), 120, dtype=np.uint8)
        class_names = ["0-25", "25-50", "50-75", "75-100"]
        event_ts = float(int(time.time()))
        capture_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event_ts))
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=event_ts,
            boxes=np.array([[28.0, 28.0, 52.0, 52.0]], dtype=np.float32),
            classes=np.array([1], dtype=np.int64),
            scores=np.array([0.61], dtype=np.float32),
            class_names=class_names,
        )
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=event_ts + 10.0,
            boxes=np.array([[28.0, 28.0, 52.0, 52.0]], dtype=np.float32),
            classes=np.array([2], dtype=np.int64),
            scores=np.array([0.61], dtype=np.float32),
            class_names=class_names,
        )

        manager = self._manager(wheel_provider=cache)
        payload = manager._build_api_payload(self._type5_event(capture_time=capture_time), self._track_state(), frame_idx=30)

        self.assertNotIn("wheelResults", payload)

    def test_type5_local_event_json_also_attaches_wheel_results(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((80, 80, 3), 120, dtype=np.uint8)
        now_ts = time.time()
        capture_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=now_ts,
            boxes=np.array([[28.0, 28.0, 52.0, 52.0]], dtype=np.float32),
            classes=np.array([2], dtype=np.int64),
            scores=np.array([0.88], dtype=np.float32),
            class_names=["0-25", "25-50", "50-75", "75-100"],
        )

        manager = self._manager(wheel_provider=cache)
        manager.enable_event_disk = True
        track_state = self._track_state()
        manager._update_track_wheel_results(1, track_state, frame_ts=now_ts)
        track_state["type1_capture_time"] = "2026-04-28 11:59:47"
        manager._emit_event_core(
            track_id=1,
            event_type=5,
            frame_idx=30,
            frame=frame,
            payload={"captureTime": capture_time, "washDuration": 7.3},
            track_state=track_state,
            vehicle_type="car",
        )

        saved = sorted(Path(manager.events_dir).glob("*_t5_*.json"))
        self.assertEqual(len(saved), 1)
        event = json.load(saved[0].open("r", encoding="utf-8"))

        self.assertIn("wheelResults", event)
        self.assertEqual(len(event["wheelResults"]), 1)
        self.assertEqual(
            set(event["wheelResults"][0].keys()),
            {"side", "captureTime", "photoUrl", "className"},
        )
        self.assertEqual(event["wheelResults"][0]["side"], "left")
        self.assertEqual(event["wheelResults"][0]["className"], "50-75")
        self.assertTrue(Path(event["wheelResults"][0]["photoUrl"]).is_absolute())
        self.assertTrue(Path(event["wheelResults"][0]["photoUrl"]).exists())

    def test_lifecycle_locked_wheel_results_survive_after_provider_no_longer_has_recent_items(self):
        provider = _StaticWheelProvider([
            {
                "entryId": 1,
                "side": "left",
                "captureTime": "2026-04-28 11:59:40",
                "imageJpegBytes": b"abc",
                "className": "25-50",
                "score": 0.71,
                "centerDistance": 12.0,
                "capture_ts": 1000.0,
            }
        ])
        manager = self._manager(wheel_provider=provider)
        track_state = self._track_state()

        manager._update_track_wheel_results(1, track_state, frame_ts=1000.0)
        provider.items = []

        payload = manager._build_api_payload(self._type5_event(capture_time="2026-04-28 12:10:00"), track_state, frame_idx=30)

        self.assertIn("wheelResults", payload)
        self.assertEqual(payload["wheelResults"][0]["className"], "25-50")

    def test_lifecycle_locked_wheel_results_upgrade_to_better_candidate(self):
        provider = _StaticWheelProvider([
            {
                "entryId": 1,
                "side": "left",
                "captureTime": "2026-04-28 11:59:40",
                "imageJpegBytes": b"first",
                "className": "0-25",
                "score": 0.95,
                "capture_ts": 1000.0,
            }
        ])
        manager = self._manager(wheel_provider=provider)
        track_state = self._track_state()

        manager._update_track_wheel_results(1, track_state, frame_ts=1000.0)
        provider.items = [
            {
                "entryId": 2,
                "side": "left",
                "captureTime": "2026-04-28 11:59:45",
                "imageJpegBytes": b"second",
                "className": "50-75",
                "score": 0.62,
                "capture_ts": 1005.0,
            }
        ]
        manager._update_track_wheel_results(1, track_state, frame_ts=1005.0)

        locked = track_state.get("wheel_results_locked", {}).get("left", {})
        self.assertEqual(locked.get("className"), "0-25")
        self.assertEqual(locked.get("imageJpegBytes"), b"first")

    def test_type5_payload_omits_wheel_results_when_provider_absent(self):
        manager = self._manager(wheel_provider=None)

        payload = manager._build_api_payload(self._type5_event(), self._track_state(), frame_idx=30)

        self.assertNotIn("wheelResults", payload)

    def test_update_track_starts_lifecycle_locking_before_type5(self):
        now_ts = time.time()
        provider = _StaticWheelProvider([
            {
                "entryId": 1,
                "side": "left",
                "captureTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts)),
                "imageJpegBytes": b"abc",
                "className": "25-50",
                "score": 0.71,
                "centerDistance": 12.0,
                "capture_ts": now_ts,
            }
        ])
        manager = self._manager(wheel_provider=provider, zone_manager=_ZoneAZoneManager())
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        manager.update_track(
            track_id=1,
            plate_box=None,
            vehicle_box=[0, 0, 20, 20],
            plate_text="",
            frame_idx=1,
            frame=frame,
            water_boxes=[],
            water_active=False,
            is_plate=False,
            vehicle_label="car",
            vehicle_conf=0.9,
            plate_conf=0.0,
            confirmed=False,
        )

        locked = manager.tracks[1].get("wheel_results_locked", {}).get("left", {})
        self.assertEqual(locked.get("className"), "25-50")

    def test_update_track_does_not_lock_before_zone_a(self):
        now_ts = time.time()
        provider = _StaticWheelProvider([
            {
                "entryId": 1,
                "side": "left",
                "captureTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts)),
                "imageJpegBytes": b"abc",
                "className": "25-50",
                "score": 0.71,
                "centerDistance": 12.0,
                "capture_ts": now_ts,
            }
        ])
        manager = self._manager(wheel_provider=provider)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        manager.update_track(
            track_id=1,
            plate_box=None,
            vehicle_box=[0, 0, 20, 20],
            plate_text="",
            frame_idx=1,
            frame=frame,
            water_boxes=[],
            water_active=False,
            is_plate=False,
            vehicle_label="car",
            vehicle_conf=0.9,
            plate_conf=0.0,
            confirmed=False,
        )

        self.assertNotIn("left", manager.tracks[1].get("wheel_results_locked", {}))

    def test_rejects_wheel_result_too_far_before_track_activation(self):
        provider = _StaticWheelProvider([
            {
                "entryId": 1,
                "side": "left",
                "captureTime": "2026-04-28 11:59:40",
                "imageJpegBytes": b"abc",
                "className": "25-50",
                "score": 0.71,
                "centerDistance": 12.0,
                "capture_ts": 1000.0,
            }
        ])
        manager = self._manager(wheel_provider=provider)
        track_state = self._track_state()
        manager._update_wheel_track_activity(1, track_state, frame_ts=1020.0, active=True)

        manager._update_track_wheel_results(1, track_state, frame_ts=1020.0)

        self.assertNotIn("left", track_state.get("wheel_results_locked", {}))

    def test_wheel_result_is_not_reused_by_other_track_after_claim(self):
        provider = _StaticWheelProvider([
            {
                "entryId": 1,
                "side": "left",
                "captureTime": "2026-04-28 11:59:40",
                "imageJpegBytes": b"abc",
                "className": "25-50",
                "score": 0.71,
                "centerDistance": 12.0,
                "capture_ts": 1000.0,
            }
        ])
        manager = self._manager(wheel_provider=provider)
        track_state_1 = self._track_state()
        track_state_2 = self._track_state()

        manager._update_track_wheel_results(1, track_state_1, frame_ts=1000.0)
        manager._update_track_wheel_results(2, track_state_2, frame_ts=1002.0)

        self.assertIn("left", track_state_1.get("wheel_results_locked", {}))
        self.assertNotIn("left", track_state_2.get("wheel_results_locked", {}))

        manager._update_track_wheel_results(2, track_state_2, frame_ts=1006.0)
        self.assertNotIn("left", track_state_2.get("wheel_results_locked", {}))

    def test_claiming_one_entry_also_claims_same_side_neighbor_entries_within_1_5_seconds(self):
        provider = _StaticWheelProvider([
            {
                "entryId": 106,
                "side": "left",
                "captureTime": "2026-04-28 11:59:40",
                "imageJpegBytes": b"a",
                "className": "0-25",
                "score": 0.60,
                "centerDistance": 8.0,
                "capture_ts": 1000.0,
            },
            {
                "entryId": 107,
                "side": "left",
                "captureTime": "2026-04-28 11:59:40",
                "imageJpegBytes": b"b",
                "className": "25-50",
                "score": 0.80,
                "centerDistance": 3.0,
                "capture_ts": 1000.6,
            },
            {
                "entryId": 108,
                "side": "left",
                "captureTime": "2026-04-28 11:59:41",
                "imageJpegBytes": b"c",
                "className": "50-75",
                "score": 0.70,
                "centerDistance": 4.0,
                "capture_ts": 1001.2,
            },
        ])
        manager = self._manager(wheel_provider=provider)
        track_state_1 = self._track_state()
        track_state_2 = self._track_state()

        manager._update_track_wheel_results(1, track_state_1, frame_ts=1001.2)
        left_locked = track_state_1.get("wheel_results_locked", {}).get("left", {})

        self.assertEqual(left_locked.get("entryId"), 107)
        self.assertEqual(provider.claims.get(106), 1)
        self.assertEqual(provider.claims.get(107), 1)
        self.assertEqual(provider.claims.get(108), 1)

        manager._update_track_wheel_results(2, track_state_2, frame_ts=1001.2)
        self.assertNotIn("left", track_state_2.get("wheel_results_locked", {}))

    def test_chain_cluster_claim_reaches_entries_beyond_direct_target_radius(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80, entry_cluster_seconds=1.5)
        frame = np.full((80, 80, 3), 120, dtype=np.uint8)
        class_names = ["0-25", "25-50", "50-75", "75-100"]

        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1000.0,
            boxes=np.array([[30.0, 30.0, 50.0, 50.0]], dtype=np.float32),
            classes=np.array([0], dtype=np.int64),
            scores=np.array([0.60], dtype=np.float32),
            class_names=class_names,
        )
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1001.4,
            boxes=np.array([[34.0, 34.0, 66.0, 66.0]], dtype=np.float32),
            classes=np.array([1], dtype=np.int64),
            scores=np.array([0.95], dtype=np.float32),
            class_names=class_names,
        )
        cache.update_from_detections(
            side="left",
            frame=frame,
            capture_ts=1002.8,
            boxes=np.array([[33.0, 33.0, 61.0, 61.0]], dtype=np.float32),
            classes=np.array([2], dtype=np.int64),
            scores=np.array([0.80], dtype=np.float32),
            class_names=class_names,
        )

        entries = cache.get_recent_result_entries(now_ts=1002.8, reference_ts=1002.8, track_id=1)
        chosen = next(item for item in entries if item["side"] == "left")
        self.assertTrue(cache.claim_result_entry(1, chosen["entryId"]))

        entries_track_2 = cache.get_recent_result_entries(now_ts=1002.8, reference_ts=1002.8, track_id=2)
        left_entries_track_2 = [item for item in entries_track_2 if item["side"] == "left"]
        self.assertEqual(left_entries_track_2, [])

    def test_runtime_config_includes_wheel_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "system": {"device_id": "wheel-test"},
                        "video": {"source": "demo.mp4"},
                        "logic": {"lane_name": "lane"},
                        "zones": {
                            "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                            "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                            "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = load_config(str(path))

        self.assertIn("wheel", config)
        self.assertFalse(config["wheel"]["enabled"])
        self.assertEqual(config["wheel"]["classes"], ["0-25", "25-50", "50-75", "75-100"])
        self.assertEqual(config["wheel"]["target_fps"], 5.0)
        self.assertEqual(config["wheel"]["reader_stale_seconds"], 5.0)
        self.assertEqual(config["wheel"]["reader_stale_check_interval_frames"], 15)
        self.assertEqual(config["wheel"]["reader_stale_hash_size"], 16)
        self.assertNotIn("center_min_margin_ratio", config["wheel"])

    def test_wheel_model_path_prefers_project_root_when_relative_path_exists(self):
        settings = resolve_wheel_settings(
            {"wheel": {"model": "models/wheel/2026.4.28CRwheel.rknn"}},
            base_dir=Path(__file__).resolve().parent,
        )

        model_path = Path(settings["model_path"])
        self.assertEqual(model_path.name, "2026.4.28CRwheel.rknn")
        self.assertEqual(model_path.parent, Path(__file__).resolve().parent.parent / "models" / "wheel")
        self.assertTrue(model_path.exists())

    def test_event_json_disk_retention_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_dir = root / "events"
            capture_dir = root / "captures"
            manager = EventManager(
                {
                    "logic": {"enable_event_disk": False},
                    "event_output_dir": str(event_dir),
                    "event_capture_dir": str(capture_dir),
                },
                fps=25.0,
                frame_size=(64, 64),
                zone_manager=_DummyZoneManager(),
            )
            track_state = {
                "plate_conf_history": [0.9],
                "vehicle_conf_history": [0.8],
                "events": set(),
                "last_frame_idx": 1,
            }

            manager._emit_event_core(
                1,
                1,
                1,
                np.zeros((64, 64, 3), dtype=np.uint8),
                {"captureTime": "2026-07-11 16:00:00"},
                track_state,
                "car",
            )

            self.assertFalse(list(event_dir.glob("*.json")))
            self.assertTrue(list(capture_dir.rglob("*.jpg")))

    def test_event_json_disk_retention_can_be_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_dir = root / "events"
            capture_dir = root / "captures"
            manager = EventManager(
                {
                    "logic": {"enable_event_disk": True},
                    "event_output_dir": str(event_dir),
                    "event_capture_dir": str(capture_dir),
                },
                fps=25.0,
                frame_size=(64, 64),
                zone_manager=_DummyZoneManager(),
            )
            track_state = {
                "plate_conf_history": [0.9],
                "vehicle_conf_history": [0.8],
                "events": set(),
                "last_frame_idx": 1,
            }

            manager._emit_event_core(
                1,
                1,
                1,
                np.zeros((64, 64, 3), dtype=np.uint8),
                {"captureTime": "2026-07-11 16:00:00"},
                track_state,
                "car",
            )

            self.assertTrue(list(event_dir.glob("*.json")))
            self.assertTrue(list(capture_dir.rglob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
