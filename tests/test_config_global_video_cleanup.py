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
            self.assertEqual(manager.logic["per_id_raw_prebuffer_seconds"], 3.0)

    def test_web_config_registry_exposes_raw_per_id_video_policy(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("logic.per_id_video_source", field_paths)
        self.assertIn("logic.per_id_raw_prebuffer_seconds", field_paths)

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
