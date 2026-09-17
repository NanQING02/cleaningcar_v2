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
                "logic": {"zone_a_mask_enable": True},
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
            self.assertEqual(manager.video["decode_backend"], "auto")
            self.assertEqual(manager.data["event_capture_quality"], 70)

            self.assertEqual(manager.video["reader_frame_timeout_seconds"], 5.0)
            self.assertEqual(manager.logic["track_lost_grace_seconds"], 4.0)
            self.assertNotIn("type34_min_interval_frames", manager.logic)
            self.assertNotIn("water_window_min_hits", manager.logic)
            self.assertNotIn("allowed_event_types", manager.logic)
            self.assertNotIn("stationary_min_frames", manager.logic)
            self.assertNotIn("stationary_speed_thresh", manager.logic)
            self.assertNotIn("water_window_size", manager.logic)
            self.assertNotIn("require_vehicle_type_for_events", manager.logic)
            self.assertNotIn("vehicle_shrink_ratio", manager.logic)
            self.assertNotIn("vehicle_lock_min_votes", manager.logic)
            self.assertNotIn("vehicle_lock_on_confirm", manager.logic)
            self.assertFalse(manager.logic["event_trace_enabled"])
            self.assertEqual(manager.logic["event_trace_dir"], "event_traces")
            self.assertEqual(manager.logic["event_trace_queue_size"], 4096)
            self.assertEqual(manager.logic["anchor_mode"], "neutral")
            self.assertEqual(manager.zones["direction_reference_edge"], "auto")
            self.assertFalse(manager.logic["anchor_shadow_compare"])
            self.assertEqual(manager.logic["anchor_legacy_flow_shift_ratio"], 0.3)
            self.assertEqual(manager.logic["anchor_adaptive_vertical_ratio"], 0.08)
            self.assertEqual(manager.logic["anchor_adaptive_vertical_cap_ratio"], 0.02)
            self.assertEqual(manager.logic["anchor_edge_margin_ratio"], 0.01)
            self.assertEqual(manager.logic["anchor_history_size"], 20)
            self.assertEqual(manager.logic["anchor_reuse_max_frames"], 12)
            self.assertEqual(manager.logic["anchor_direction_window"], 8)
            self.assertEqual(manager.logic["anchor_direction_min_points"], 5)
            self.assertEqual(manager.logic["anchor_direction_consistency"], 0.7)
            self.assertEqual(manager.logic["anchor_direction_min_displacement_ratio"], 0.03)
            self.assertEqual(manager.logic["anchor_direction_blend_frames"], 5)
            self.assertNotIn("zone_a_mask_enable", manager.logic)
            self.assertEqual(manager.logic["zone_a_margin_ratio"], 0.10)
            self.assertEqual(manager.logic["zone_a_margin_min_px"], 4.0)
            self.assertEqual(manager.logic["zone_a_margin_max_px"], 24.0)
            self.assertEqual(manager.logic["zone_a_observed_outside_hits"], 3)
            self.assertEqual(manager.logic["zone_a_enter_core_hits"], 3)
            self.assertEqual(manager.logic["zone_a_exit_outside_hits"], 5)
            self.assertFalse(manager.data["wheel"]["pause_bypass_during_wash_enabled"])
            self.assertEqual(manager.data["wheel"]["pause_bypass_config_key"], "config_绕行.json")
            self.assertEqual(manager.data["wheel"]["pause_bypass_resume_delay_seconds"], 0.5)
            self.assertEqual(manager.data["event_capture_dir"], "/data/ftp/event_captures/config")

    def test_web_config_registry_exposes_reader_frame_timeout(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("video.reader_frame_timeout_seconds", field_paths)
        self.assertIn("logic.track_lost_grace_seconds", field_paths)
        self.assertNotIn("logic.type34_min_interval_frames", field_paths)
        self.assertNotIn("logic.allowed_event_types", field_paths)
        self.assertIn("logic.event_trace_enabled", field_paths)
        self.assertIn("logic.event_trace_dir", field_paths)
        self.assertIn("logic.event_trace_queue_size", field_paths)
        self.assertNotIn("logic.anchor_mode", field_paths)
        self.assertIn("zones.direction_reference_edge", field_paths)
        self.assertIn("logic.anchor_shadow_compare", field_paths)
        self.assertIn("logic.anchor_direction_window", field_paths)
        self.assertIn("logic.anchor_direction_min_points", field_paths)
        self.assertIn("logic.anchor_direction_consistency", field_paths)
        self.assertIn("logic.anchor_direction_min_displacement_ratio", field_paths)
        self.assertIn("logic.anchor_direction_blend_frames", field_paths)
        self.assertNotIn("logic.zone_a_mask_enable", field_paths)
        for path in (
            "logic.zone_a_margin_ratio",
            "logic.zone_a_margin_min_px",
            "logic.zone_a_margin_max_px",
            "logic.zone_a_observed_outside_hits",
            "logic.zone_a_enter_core_hits",
            "logic.zone_a_exit_outside_hits",
        ):
            self.assertNotIn(path, field_paths)

    def test_config_manager_defaults_simplified_event_controls(self):
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

            self.assertEqual(manager.logic["plate_track_lock_frames"], 6)
            self.assertEqual(manager.logic["event_plate_lock_frames"], 6)
            self.assertEqual(manager.logic["event_plate_fast_lock_frames"], 3)
            self.assertTrue(manager.logic["plate_output_shape_log_once"])
            self.assertEqual(manager.logic["shadow_plate_pool"]["text_window_frames"], 50)
            self.assertEqual(manager.logic["shadow_plate_pool"]["color_min_confidence"], 0.70)
            self.assertEqual(manager.logic["min_track_frames_for_type1"], 5)
            self.assertEqual(manager.logic["water_confirm_frames"], 3)
            self.assertEqual(manager.logic["min_zone_b_dwell_seconds_for_type4"], 0.5)
            self.assertNotIn("event_track_quality", manager.logic)
            self.assertNotIn("min_zone_b_dwell_frames_for_type4", manager.logic)

    def test_web_config_registry_exposes_simplified_event_controls(self):
        field_paths = {item["path"] for item in CONFIG_FIELD_REGISTRY}

        self.assertIn("logic.plate_output_shape_log_once", field_paths)
        self.assertIn("logic.shadow_plate_pool.text_window_frames", field_paths)
        self.assertIn("logic.shadow_plate_pool.color_min_confidence", field_paths)
        self.assertIn("logic.min_track_frames_for_type1", field_paths)
        self.assertIn("logic.water_confirm_frames", field_paths)
        self.assertIn("logic.min_zone_b_dwell_seconds_for_type4", field_paths)
        self.assertNotIn("logic.event_track_quality.min_hits_type1", field_paths)
        self.assertNotIn("logic.event_track_quality.suppress_obvious_false_type5", field_paths)

    def test_config_manager_removes_legacy_event_quality_value(self):
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

            self.assertNotIn("event_track_quality", manager.logic)

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

            self.assertEqual(manager.video["decode_backend"], "auto")

    def test_config_manager_forces_rtsp_to_gstreamer_direct_bgr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {
                    "source": "rtsp://camera/live",
                    "source_mode": "camera",
                    "decode_backend": "ffmpeg",
                    "gstreamer_bgr_mode": "safe",
                },
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertEqual(manager.video["decode_backend"], "gstreamer")
            self.assertEqual(manager.video["gstreamer_bgr_mode"], "direct")

    def test_existing_local_file_respects_explicit_ffmpeg_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            video_path.write_bytes(b"placeholder")
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {
                    "source": str(video_path),
                    "source_mode": "camera",
                    "decode_backend": "ffmpeg",
                },
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertEqual(manager.video["decode_backend"], "ffmpeg")

    def test_config_manager_strips_ffmpeg_rga_backend_and_section(self):
        """RGA 管控（2026-08-26）：ffmpeg_rga 后端已删除；残留配置必须归一到 auto 并丢弃 ffmpeg_rga 段。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {
                "system": {"device_id": "cam-a"},
                "video": {
                    "source": "demo.mp4",
                    "decode_backend": "ffmpeg_rga",
                    "ffmpeg_rga": {
                        "core": "rga3_core1",
                        "width": 640,
                        "height": 360,
                        "async_depth": 9,
                        "breaker_enabled": True,
                    },
                },
                "zones": {
                    "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                },
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manager = ConfigManager(path)

            self.assertEqual(manager.video["decode_backend"], "auto")
            self.assertNotIn("ffmpeg_rga", manager.video)

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
