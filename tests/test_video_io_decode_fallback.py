import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cleaningcar import video_io


class _DummyCapture:
    def __init__(self, opened):
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        return True, "frame"

    def release(self):
        self.released = True
        self._opened = False
        return None


class VideoIoDecodeFallbackTests(unittest.TestCase):
    @staticmethod
    def _args(hw_decode=True):
        return SimpleNamespace(
            hw_decode=hw_decode,
            _config={
                "video": {
                    "rtsp_latency_ms": 200,
                    "rtsp_appsink_max_buffers": 1,
                    "reader_frame_timeout_seconds": 7.5,
                }
            },
        )

    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_prefers_gstreamer_direct_bgr_hardware_decode(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
    ):
        gstreamer_hw_open.return_value = _DummyCapture(True)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNotNone(cap)
        self.assertEqual(meta["decode_mode"], "hw")
        self.assertEqual(meta["decode_backend"], "gstreamer")
        self.assertFalse(meta["fallback_used"])
        self.assertEqual(meta["source_kind"], "rtsp")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw"])
        self.assertEqual(meta["reader_frame_timeout_seconds"], 7.5)
        self.assertEqual(gstreamer_hw_open.call_args.kwargs["read_timeout_seconds"], 7.5)
        self.assertEqual(gstreamer_hw_open.call_args.kwargs["bgr_mode"], "direct")
        ffmpeg_hw_open.assert_not_called()

    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_uses_ffmpeg_hardware_after_gstreamer_failure(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
    ):
        gstreamer_hw_open.return_value = _DummyCapture(False)
        ffmpeg_hw_open.return_value = _DummyCapture(True)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNotNone(cap)
        self.assertEqual(meta["decode_mode"], "hw")
        self.assertEqual(meta["decode_backend"], "ffmpeg")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "gstreamer_hw_open_failed")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw", "ffmpeg_hw"])

    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_reports_failure_after_both_hardware_paths_fail(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
    ):
        ffmpeg_hw_open.return_value = _DummyCapture(False)
        gstreamer_hw_open.return_value = _DummyCapture(False)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNone(cap)
        self.assertEqual(meta["decode_mode"], "none")
        self.assertEqual(meta["decode_backend"], "none")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "hardware_open_failed")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw", "ffmpeg_hw"])

    def test_ffmpeg_rawvideo_timeout_returns_failure_and_records_diagnostics(self):
        cap = video_io.FfmpegRawVideoCapture.__new__(video_io.FfmpegRawVideoCapture)
        cap.src = "rtsp://camera"
        cap.decoder = "hevc_rkmpp"
        cap.width = 1920
        cap.height = 1080
        cap.fps = 25.0
        cap.codec_name = "hevc"
        cap.frame_bytes = 16
        cap.read_timeout_seconds = 3.0
        cap.backend = "ffmpeg_rawvideo"
        cap.proc = object()
        cap._opened = True
        cap._stderr_lines = []
        cap._recent_error_lines = []
        cap._recent_error_match_count = 0
        cap._last_read_error = ""
        cap._last_read_error_ts = 0.0
        cap.release = unittest.mock.Mock(side_effect=lambda: setattr(cap, "_opened", False))
        cap.isOpened = lambda: True
        cap._read_exact_with_timeout = unittest.mock.Mock(side_effect=TimeoutError("ffmpeg_rawvideo_read_timeout"))

        ok, frame = video_io.FfmpegRawVideoCapture.read(cap)

        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertEqual(cap._last_read_error, "ffmpeg_rawvideo_read_timeout")
        self.assertGreater(cap._last_read_error_ts, 0.0)
        cap.release.assert_called_once()

    @patch("cleaningcar.video_io._open_gstreamer_hardware_capture")
    @patch("cleaningcar.video_io._open_ffmpeg_hardware_capture")
    def test_create_video_reader_reports_total_failure_after_all_fallbacks(
        self,
        ffmpeg_hw_open,
        gstreamer_hw_open,
    ):
        ffmpeg_hw_open.return_value = _DummyCapture(False)
        gstreamer_hw_open.return_value = _DummyCapture(False)

        cap, meta = video_io.create_video_reader("rtsp://camera", self._args(hw_decode=True))

        self.assertIsNone(cap)
        self.assertEqual(meta["decode_mode"], "none")
        self.assertEqual(meta["decode_backend"], "none")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "hardware_open_failed")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw", "ffmpeg_hw"])

    def test_timed_video_capture_timeout_returns_failure_and_releases_capture(self):
        class _BlockingCapture(_DummyCapture):
            def read(self):
                import time

                time.sleep(0.2)
                return True, "late"

        inner = _BlockingCapture(True)
        cap = video_io.TimedVideoCapture(inner, read_timeout_seconds=0.01, backend="gstreamer_hw")

        ok, frame = cap.read()

        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertTrue(inner.released)
        self.assertIn("timed_capture_read_timeout", cap.diagnostics()["last_read_error"])


if __name__ == "__main__":
    unittest.main()
