import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from future_modules.acceleration import rga_resize_plugin


class RgaResizePluginTests(unittest.TestCase):
    def test_bgr_stride_aligns_to_16_bytes(self):
        self.assertEqual(rga_resize_plugin._align_bgr888_stride(76), 80)
        self.assertEqual(rga_resize_plugin._align_bgr888_stride(84), 96)
        self.assertEqual(rga_resize_plugin._align_bgr888_stride(128), 128)

    def test_small_resize_falls_back_to_cv2_before_touching_librga(self):
        image = np.ones((34, 64, 3), dtype=np.uint8)
        fallback = np.zeros((48, 176, 3), dtype=np.uint8)

        with patch.object(rga_resize_plugin, "RGA_OK", True), \
                patch.object(rga_resize_plugin, "_lib", object()), \
                patch.object(
                    rga_resize_plugin,
                    "_resize_with_librga",
                    side_effect=AssertionError("should not call librga"),
                ), \
                patch("future_modules.acceleration.rga_resize_plugin.cv2.resize", return_value=fallback) as mock_resize:
            result = rga_resize_plugin.rga_resize(image, (176, 48))

        self.assertIs(result, fallback)
        mock_resize.assert_called_once()

    def test_rga_calls_are_serialized_by_process_lock(self):
        image = np.ones((128, 128, 3), dtype=np.uint8)
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_resize(src, dst):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            dst[:] = 7
            with lock:
                active -= 1
            return 1

        with patch.object(rga_resize_plugin, "RGA_OK", True), \
                patch.object(rga_resize_plugin, "_lib", object()), \
                patch.object(rga_resize_plugin, "_resize_with_librga", side_effect=fake_resize):
            threads = [
                threading.Thread(target=rga_resize_plugin.rga_resize, args=(image, (256, 128)))
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
