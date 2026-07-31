import unittest
from unittest.mock import patch

from cleaningcar import video_io


class _DummyWriter:
    def __init__(self, opened):
        self._opened = opened
        self.released = False

    def is_opened(self):
        return self._opened

    def release(self):
        self.released = True
        return False


class VideoIoWriterFallbackTests(unittest.TestCase):
    @patch("cleaningcar.video_io.FfmpegH264Writer")
    def test_create_h264_video_writer_uses_only_ffmpeg_hardware_writer(self, ffmpeg_writer_cls):
        ffmpeg_writer = _DummyWriter(True)
        ffmpeg_writer_cls.return_value = ffmpeg_writer

        writer, meta = video_io.create_h264_video_writer("demo.mp4", 1920, 1080, 25.0)

        self.assertIs(writer, ffmpeg_writer)
        self.assertEqual(meta["writer_mode"], "hw")
        self.assertEqual(meta["writer_backend"], "ffmpeg")
        self.assertEqual(meta["attempt_order"], ["ffmpeg_hw"])
        self.assertFalse(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "")
        ffmpeg_writer_cls.assert_called_once_with(
            "demo.mp4",
            1920,
            1080,
            25.0,
            encoders=video_io.FFMPEG_HW_ENCODERS,
        )

    @patch("cleaningcar.video_io.FfmpegH264Writer")
    def test_create_h264_video_writer_reports_ffmpeg_failure(self, ffmpeg_writer_cls):
        ffmpeg_hw_writer = _DummyWriter(False)
        ffmpeg_writer_cls.return_value = ffmpeg_hw_writer

        writer, meta = video_io.create_h264_video_writer("demo.mp4", 1920, 1080, 25.0)

        self.assertIsNone(writer)
        self.assertEqual(meta["writer_mode"], "none")
        self.assertEqual(meta["writer_backend"], "none")
        self.assertFalse(meta["fallback_used"])
        self.assertEqual(meta["fallback_reason"], "ffmpeg_hw_open_failed")
        self.assertEqual(meta["attempt_order"], ["ffmpeg_hw"])
        self.assertTrue(ffmpeg_hw_writer.released)


if __name__ == "__main__":
    unittest.main()
