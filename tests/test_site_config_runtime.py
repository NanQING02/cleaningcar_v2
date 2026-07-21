import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SiteConfigRuntimeTests(unittest.TestCase):
    def test_main_and_bypass_plate_stride_match_runtime_configs(self):
        expected_plate_stride = {
            "configs/config.json": 2,
            "configs/config_绕行.json": 2,
        }
        for relative, plate_stride in expected_plate_stride.items():
            with self.subTest(config=relative):
                data = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
                self.assertEqual(data["video"]["decode_backend"], "ffmpeg")
                self.assertEqual(data["logic"]["plate_infer_stride"], plate_stride)
                self.assertNotIn("run_mode", data["wheel"])
                self.assertNotIn("service_url", data["wheel"])
                self.assertNotIn("service_timeout_seconds", data["wheel"])

    def test_site_npu_core_assignment_is_fixed(self):
        main = json.loads((PROJECT_ROOT / "configs/config.json").read_text(encoding="utf-8"))
        bypass = json.loads((PROJECT_ROOT / "configs/config_绕行.json").read_text(encoding="utf-8"))

        self.assertEqual(main["video"]["workers"], 1)
        self.assertEqual(main["video"]["core_mask"], "0")
        self.assertEqual(main["logic"]["plate_core_mask"], "0")
        self.assertEqual(main["wheel"]["core_mask"], "2")

        self.assertEqual(bypass["video"]["workers"], 1)
        self.assertEqual(bypass["video"]["core_mask"], "1")
        self.assertEqual(bypass["logic"]["plate_core_mask"], "1")
        self.assertFalse(bypass["logic"]["zone_a_mask_enable"])
        self.assertFalse(bypass["logic"]["plate_requires_vehicle"])
        self.assertTrue(bypass["logic"]["disable_plate_only_events"])
        self.assertEqual(bypass["logic"]["pending_plate_cache_ttl_frames"], 40)
        self.assertFalse(bypass["wheel"]["enabled"])

    def test_main_and_bypass_configs_are_isolated(self):
        main = json.loads((PROJECT_ROOT / "configs/config.json").read_text(encoding="utf-8"))
        bypass = json.loads((PROJECT_ROOT / "configs/config_绕行.json").read_text(encoding="utf-8"))

        self.assertNotEqual(main["system"]["device_id"], bypass["system"]["device_id"])
        self.assertNotEqual(main["video"]["source"], bypass["video"]["source"])
        self.assertNotEqual(main["video"]["core_mask"], bypass["video"]["core_mask"])
        self.assertNotEqual(main["logic"]["plate_core_mask"], bypass["logic"]["plate_core_mask"])
        self.assertNotEqual(main["event_output_dir"], bypass["event_output_dir"])
        self.assertNotEqual(main["event_capture_dir"], bypass["event_capture_dir"])
        self.assertTrue(main["wheel"]["enabled"])
        self.assertFalse(bypass["wheel"]["enabled"])
        self.assertEqual(main["wheel"]["pause_bypass_config_key"], "config_绕行.json")
        self.assertIsNone(main["logic"].get("plate_requires_vehicle"))
        self.assertFalse(bypass["logic"]["plate_requires_vehicle"])

    def test_site_configs_enable_performance_lock(self):
        for relative in ("configs/config.json", "configs/config_绕行.json"):
            with self.subTest(config=relative):
                data = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
                self.assertTrue(data["system"]["performance_lock_enabled"])


if __name__ == "__main__":
    unittest.main()
