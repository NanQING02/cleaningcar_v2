import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from web import server, state


class WebConfigNamespaceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.config_dir = self.base / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.original_config_path = state.CONFIG_PATH

    def tearDown(self):
        state.set_config_path(self.original_config_path)

    def write_config(self, name: str, device_id: str, lane_name: str = "") -> Path:
        payload = {
            "system": {"device_id": device_id},
            "logic": {"lane_name": lane_name or name},
            "video": {"source": "demo.mp4"},
            "zones": {
                "zone_a_detection": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                "zone_b_wash": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                "flow_vector": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
            },
        }
        path = self.config_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_validate_device_id_unique_rejects_duplicates(self):
        target = self.write_config("config_main.json", "camera-a")
        self.write_config("config_other.json", "camera-a")
        with self.assertRaises(HTTPException) as ctx:
            server._validate_device_id_unique(target, {"system": {"device_id": "camera-a"}})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("device_id", str(ctx.exception.detail))

    def test_list_config_files_returns_device_ids(self):
        active = self.write_config("config.json", "camera-a")
        self.write_config("config_b.json", "camera-b")
        self.write_config("client_lane.json", "camera-c")
        state.set_config_path(active)

        payload = server.list_config_files()

        self.assertEqual(payload["active"], "config.json")
        self.assertIn("client_lane.json", payload["files"])
        self.assertEqual(payload["device_ids"]["config.json"], "camera-a")
        self.assertEqual(payload["device_ids"]["config_b.json"], "camera-b")
        self.assertEqual(payload["device_ids"]["client_lane.json"], "camera-c")

    def test_read_config_uses_key(self):
        active = self.write_config("config.json", "camera-a", lane_name="A")
        self.write_config("config_b.json", "camera-b", lane_name="B")
        state.set_config_path(active)

        payload = server.read_config("config_b.json")

        self.assertEqual(payload["system"]["device_id"], "camera-b")
        self.assertEqual(payload["logic"]["lane_name"], "B")

    def test_debug_frame_meta_distinguishes_disabled_and_pending_file(self):
        active = self.write_config("config.json", "camera-a", lane_name="A")
        state.set_config_path(active)

        cfg = server._load_config()
        cfg.data.setdefault("video", {})["debug_frame_path"] = "off"
        cfg.save()
        payload = server.debug_frame_meta()
        self.assertFalse(payload["available"])
        self.assertFalse(payload["enabled"])

        cfg = server._load_config()
        cfg.data.setdefault("video", {})["debug_frame_path"] = ""
        cfg.save()
        payload = server.debug_frame_meta()
        self.assertFalse(payload["available"])
        self.assertTrue(payload["enabled"])
        debug_path = Path(payload["path"])
        self.assertEqual(debug_path.name, "debug.jpg")
        self.assertEqual(debug_path.parent.name, "camera-a")
        self.assertEqual(debug_path.parent.parent.name, "cleaningcar_runtime")

    def test_zone_editor_hides_legacy_cleanup_and_global_video_controls(self):
        template_path = Path(__file__).resolve().parent.parent / "web" / "templates" / "zone_editor.html"
        text = template_path.read_text(encoding="utf-8")

        self.assertNotIn("清理策略", text)
        self.assertNotIn("全局录像输出", text)
        self.assertNotIn("save_video", text)
        self.assertNotIn("enable_global_video", text)


if __name__ == "__main__":
    unittest.main()
