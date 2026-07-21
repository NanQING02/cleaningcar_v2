import json
import tempfile
import unittest
from pathlib import Path

from config_manager import ConfigManager, _derive_wheel_photo_url
from web.config_tiers import CONFIG_FIELD_REGISTRY


class GlobalVideoCleanupTests(unittest.TestCase):
    def test_config_manager_strips_obsolete_global_video_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4", "save_video": "demo-out.mp4"},
                "logic": {"enable_global_video": True},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertNotIn("save_video", manager.video)
            self.assertNotIn("enable_global_video", manager.logic)

    def test_web_config_registry_hides_global_video_fields(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertNotIn("video.save_video", field_paths)
        self.assertNotIn("logic.enable_global_video", field_paths)

    def test_config_manager_strips_obsolete_per_id_recording_tuning_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "logic": {
                    "per_id_downscale_ratio": 0.5,
                    "per_id_frame_stride": 2,
                    "per_id_target_width": 1280,
                    "per_id_target_height": 720,
                    "per_id_auto_adapt": True,
                    "per_id_auto_cpu_high": 80,
                    "per_id_auto_cpu_low": 40,
                    "per_id_max_frame_stride": 4,
                },
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertNotIn("per_id_downscale_ratio", manager.logic)
            self.assertNotIn("per_id_frame_stride", manager.logic)
            self.assertNotIn("per_id_target_width", manager.logic)
            self.assertNotIn("per_id_target_height", manager.logic)
            self.assertNotIn("per_id_auto_adapt", manager.logic)
            self.assertNotIn("per_id_auto_cpu_high", manager.logic)
            self.assertNotIn("per_id_auto_cpu_low", manager.logic)
            self.assertNotIn("per_id_max_frame_stride", manager.logic)

    def test_web_config_registry_hides_per_id_recording_tuning_fields(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertNotIn("logic.per_id_downscale_ratio", field_paths)
        self.assertNotIn("logic.per_id_frame_stride", field_paths)
        self.assertNotIn("logic.per_id_auto_adapt", field_paths)
        self.assertNotIn("logic.per_id_auto_cpu_high", field_paths)
        self.assertNotIn("logic.per_id_auto_cpu_low", field_paths)
        self.assertNotIn("logic.per_id_max_frame_stride", field_paths)

    def test_config_manager_defaults_raw_per_id_video_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertEqual(manager.logic["per_id_video_source"], "auto")
            self.assertNotIn("per_id_raw_prebuffer_seconds", manager.logic)

    def test_web_config_registry_exposes_raw_per_id_video_policy(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("logic.per_id_video_source", field_paths)
        self.assertNotIn("logic.per_id_raw_prebuffer_seconds", field_paths)

    def test_web_config_registry_exposes_plate_prefetch_controls(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("logic.plate_requires_vehicle", field_paths)
        self.assertIn("logic.pending_plate_cache_ttl_frames", field_paths)
        self.assertIn("logic.pending_plate_cache_max_entries", field_paths)

    def test_config_manager_defaults_reader_frame_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertEqual(manager.video["debug_frame_path"], "off")
            self.assertTrue(manager.video["hw_decode"])
            self.assertEqual(manager.video["decode_backend"], "ffmpeg")
            self.assertEqual(manager.data["event_capture_quality"], 70)

            self.assertEqual(manager.video["reader_frame_timeout_seconds"], 5.0)
            self.assertFalse(manager.data["wheel"]["pause_bypass_during_wash_enabled"])
            self.assertEqual(manager.data["wheel"]["pause_bypass_config_key"], "config_绕行.json")
            self.assertEqual(manager.data["wheel"]["pause_bypass_resume_delay_seconds"], 0.5)
            self.assertEqual(manager.data["event_capture_dir"], "/data/ftp/event_captures/config")

    def test_web_config_registry_exposes_reader_frame_timeout(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("video.reader_frame_timeout_seconds", field_paths)

    def test_config_manager_defaults_plate_and_event_quality_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertEqual(manager.logic["plate_lock_frames"], 6)
            self.assertTrue(manager.logic["plate_output_shape_log_once"])
            self.assertTrue(manager.logic["plate_draw_stable_only"])
            self.assertEqual(manager.logic["shadow_plate_pool"]["text_window_frames"], 50)
            self.assertEqual(manager.logic["shadow_plate_pool"]["color_min_confidence"], 0.70)
            self.assertTrue(manager.logic["event_track_quality"]["enabled"])
            self.assertEqual(manager.logic["event_track_quality"]["min_hits_type1"], 12)
            self.assertEqual(manager.logic["event_track_quality"]["min_zone_a_dwell_type5"], 15)

    def test_web_config_registry_exposes_plate_and_event_quality_controls(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("logic.plate_output_shape_log_once", field_paths)
        self.assertIn("logic.plate_draw_stable_only", field_paths)
        self.assertIn("logic.shadow_plate_pool.text_window_frames", field_paths)
        self.assertIn("logic.shadow_plate_pool.color_min_confidence", field_paths)
        self.assertIn("logic.event_track_quality.min_hits_type1", field_paths)
        self.assertIn("logic.event_track_quality.suppress_obvious_false_type5", field_paths)

    def test_config_manager_accepts_boolean_event_quality_legacy_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "logic": {"event_track_quality": False},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertIsInstance(manager.logic["event_track_quality"], dict)
            self.assertFalse(manager.logic["event_track_quality"]["enabled"])
            self.assertEqual(manager.logic["event_track_quality"]["min_hits_type1"], 12)

    def test_config_manager_normalizes_disabled_software_decode_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4", "decode_backend": "software"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertTrue(manager.video["hw_decode"])
            self.assertEqual(manager.video["decode_backend"], "ffmpeg")

    def test_web_config_registry_exposes_wash_priority_pause_fields(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("wheel.pause_bypass_during_wash_enabled", field_paths)
        self.assertIn("wheel.pause_bypass_config_key", field_paths)
        self.assertIn("wheel.pause_bypass_resume_delay_seconds", field_paths)

    def test_config_manager_strips_remote_wheel_service_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "wheel": {
                    "enabled": True,
                    "run_mode": "remote",
                    "service_url": "http://127.0.0.1:28015",
                    "service_timeout_seconds": 0.5,
                },
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)
            manager.save()
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("run_mode", manager.data["wheel"])
        self.assertNotIn("service_url", manager.data["wheel"])
        self.assertNotIn("service_timeout_seconds", manager.data["wheel"])
        self.assertNotIn("run_mode", saved["wheel"])
        self.assertNotIn("service_url", saved["wheel"])
        self.assertNotIn("service_timeout_seconds", saved["wheel"])

    def test_web_config_registry_hides_remote_wheel_service_fields(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertNotIn("wheel.run_mode", field_paths)
        self.assertNotIn("wheel.service_url", field_paths)
        self.assertNotIn("wheel.service_timeout_seconds", field_paths)

    def test_web_config_registry_exposes_event_path_controls_as_user_fields(self):
        user_fields = {
            item["path"]
            for item in CONFIG_FIELD_REGISTRY
            if item["tier"] == "user"
        }
        developer_fields = {
            item["path"]
            for item in CONFIG_FIELD_REGISTRY
            if item["tier"] == "developer"
        }

        self.assertIn("system.api.capture_mode", user_fields)
        self.assertIn("event_capture_dir", user_fields)
        self.assertIn("logic.enable_event_disk", user_fields)
        self.assertNotIn("event_capture_dir", developer_fields)
        self.assertNotIn("logic.enable_event_disk", developer_fields)

    def test_config_manager_strips_obsolete_rga_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4", "rga_enable": True},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertNotIn("rga_enable", manager.video)

    def test_web_config_registry_hides_rga_toggle(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertNotIn("video.rga_enable", field_paths)

    def test_config_manager_enables_performance_lock_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertTrue(manager.system["performance_lock_enabled"])

    def test_web_config_registry_hides_storage_cleanup_fields(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertFalse(any(path.startswith("storage.") for path in field_paths))

    def test_config_manager_strips_storage_cleanup_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {"source": "demo.mp4"},
                "storage": {"clean_interval_seconds": 1, "capture_keep_days": 1},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertNotIn("storage", manager.data)
            self.assertEqual(manager.storage, {})

    def test_wheel_photo_url_derivation_replaces_trailing_slash_path_segment(self):
        self.assertEqual(
            _derive_wheel_photo_url("http://example.com/api/vehicle/wash-event/"),
            "http://example.com/api/vehicle/wheel-photo",
        )

    def test_config_manager_keeps_derived_wheel_photo_url_empty_on_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {
                    "device_id": "cam-a",
                    "api": {
                        "url": "http://example.com/api/vehicle/wash-event",
                        "wheel_photo_url": "",
                    },
                },
                "video": {"source": "demo.mp4"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)
            self.assertEqual(
                manager.system["api"]["wheel_photo_url"],
                "http://example.com/api/vehicle/wheel-photo",
            )
            manager.save()
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["system"]["api"]["wheel_photo_url"], "")

    def test_config_manager_preserves_explicit_wheel_photo_url_on_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            explicit_url = "http://example.com/custom/wheel-photo"
            payload = {
                "system": {
                    "device_id": "cam-a",
                    "api": {
                        "url": "http://example.com/api/vehicle/wash-event",
                        "wheel_photo_url": explicit_url,
                    },
                },
                "video": {"source": "demo.mp4"},
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)
            manager.save()
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["system"]["api"]["wheel_photo_url"], explicit_url)


if __name__ == "__main__":
    unittest.main()
