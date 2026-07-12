import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SiteConfigRuntimeTests(unittest.TestCase):
    def test_main_and_bypass_plate_stride_match_runtime_configs(self):
        for relative in ("configs/config.json", "configs/config_绕行.json"):
            with self.subTest(config=relative):
                data = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
                self.assertEqual(data["video"]["decode_backend"], "auto")
                self.assertEqual(data["logic"]["plate_infer_stride"], 1)
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
        self.assertFalse(bypass["wheel"]["enabled"])

    def test_site_configs_enable_performance_lock(self):
        for relative in ("configs/config.json", "configs/config_绕行.json"):
            with self.subTest(config=relative):
                data = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
                self.assertTrue(data["system"]["performance_lock_enabled"])


if __name__ == "__main__":
    unittest.main()
