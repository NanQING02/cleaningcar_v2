import unittest
from pathlib import Path

from cleaningcar.runtime_signals import resolve_runtime_settings


class RuntimeSignalsDebugFramePathTests(unittest.TestCase):
    def test_empty_debug_frame_path_uses_namespaced_default(self):
        cfg = {
            "system": {"device_id": "RK3588-DEV"},
            "video": {"debug_frame_path": ""},
        }

        runtime = resolve_runtime_settings(cfg, None, namespace_hint="config")

        expected = Path("/dev/shm/cleaningcar_runtime/RK3588-DEV/debug.jpg").resolve()
        self.assertEqual(runtime["debug_frame_path"], expected)

    def test_off_debug_frame_path_disables_debug_frame(self):
        cfg = {
            "system": {"device_id": "RK3588-DEV"},
            "video": {"debug_frame_path": "off"},
        }

        runtime = resolve_runtime_settings(cfg, None, namespace_hint="config")

        self.assertIsNone(runtime["debug_frame_path"])


if __name__ == "__main__":
    unittest.main()
