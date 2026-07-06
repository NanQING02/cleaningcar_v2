import unittest
from unittest.mock import patch

import numpy as np

from cleaningcar import resize_accel


class ResizeAccelRuntimeTests(unittest.TestCase):
    def test_resize_backend_name_reports_cv2_by_default(self):
        self.assertEqual(resize_accel.resize_backend_name(), "cv2")

    def test_resize_bgr_uses_cv2(self):
        image = np.ones((32, 64, 3), dtype=np.uint8)
        fallback = np.zeros((48, 96, 3), dtype=np.uint8)

        with patch("cleaningcar.resize_accel.cv2.resize", return_value=fallback) as mock_resize:
            result = resize_accel.resize_bgr(image, (96, 48))

        self.assertIs(result, fallback)
        mock_resize.assert_called_once_with(image, (96, 48), interpolation=1)


if __name__ == "__main__":
    unittest.main()
