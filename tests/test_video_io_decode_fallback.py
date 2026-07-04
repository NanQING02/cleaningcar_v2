import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cleaningcar import video_io


class _DummyCapture:
    def __init__(self, opened):
        self._opened = opened

    def isOpened(self):
        return self._opened

    def release(self):
        return None


class VideoIoDecodeFallbackTests(unittest.TestCase):
    @staticmethod
    def _args(hw_decode=True):
        return SimpleNamespace(
            hw_decode=hw_decode,
            _config={"video": {"rtsp_latency_ms": 200, "rtsp_appsink_max_buffers": 1}},
        )

    @patch("cleaningcar.video_io._open_software_capture")
    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_prefers_ffmpeg_hardware_decode(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
        software_open,
    ):
        ffmpeg_hw_open.return_value = _DummyCapture(True)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNotNone(cap)
        self.assertEqual(meta["decode_mode"], "hw")
        self.assertEqual(meta["decode_backend"], "ffmpeg")
        self.assertFalse(meta["fallback_used"])
        self.assertEqual(meta["source_kind"], "rtsp")
        self.assertEqual(meta["attempt_order"], ["ffmpeg_hw"])
        gstreamer_hw_open.assert_not_called()
        software_open.assert_not_called()

    @patch("cleaningcar.video_io._open_software_capture")
    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_uses_gstreamer_hardware_after_ffmpeg_failure(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
        software_open,
    ):
        ffmpeg_hw_open.return_value = _DummyCapture(False)
        gstreamer_hw_open.return_value = _DummyCapture(True)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNotNone(cap)
        self.assertEqual(meta["decode_mode"], "hw")
        self.assertEqual(meta["decode_backend"], "gstreamer")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "ffmpeg_hw_open_failed")
        self.assertEqual(meta["attempt_order"], ["ffmpeg_hw", "gstreamer_hw"])
        software_open.assert_not_called()

    @patch("cleaningcar.video_io._open_software_capture")
    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_falls_back_to_software_after_both_hardware_paths_fail(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
        software_open,
    ):
        ffmpeg_hw_open.return_value = _DummyCapture(False)
        gstreamer_hw_open.return_value = _DummyCapture(False)
        software_open.return_value = _DummyCapture(True)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNotNone(cap)
        self.assertEqual(meta["decode_mode"], "sw")
        self.assertEqual(meta["decode_backend"], "software")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "hw_open_failed")
        self.assertEqual(meta["attempt_order"], ["ffmpeg_hw", "gstreamer_hw", "software"])

    @patch("cleaningcar.video_io._open_software_capture")
    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_reports_total_failure_after_all_fallbacks(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
        software_open,
    ):
        ffmpeg_hw_open.return_value = _DummyCapture(False)
        gstreamer_hw_open.return_value = _DummyCapture(False)
        software_open.return_value = _DummyCapture(False)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNone(cap)
        self.assertEqual(meta["decode_mode"], "sw")
        self.assertEqual(meta["decode_backend"], "software")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "hw_open_failed")
        self.assertEqual(meta["attempt_order"], ["ffmpeg_hw", "gstreamer_hw", "software"])


if __name__ == "__main__":
    unittest.main()
