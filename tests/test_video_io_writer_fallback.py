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
    @patch("cleaningcar.video_io.GstreamerH264Writer")
    @patch("cleaningcar.video_io.FfmpegH264Writer")
    def test_create_h264_video_writer_prefers_gstreamer_hardware_writer(self, ffmpeg_writer_cls, gstreamer_writer_cls):
        gst_writer = _DummyWriter(True)
        gstreamer_writer_cls.return_value = gst_writer

        writer, meta = video_io.create_h264_video_writer("demo.mp4", 1920, 1080, 25.0)

        self.assertIs(writer, gst_writer)
        self.assertEqual(meta["writer_mode"], "hw")
        self.assertEqual(meta["writer_backend"], "gstreamer")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw"])
        gstreamer_writer_cls.assert_called_once()
        ffmpeg_writer_cls.assert_not_called()

    @patch("cleaningcar.video_io.GstreamerH264Writer")
    @patch("cleaningcar.video_io.FfmpegH264Writer")
    def test_create_h264_video_writer_uses_ffmpeg_hardware_writer_after_gstreamer_failure(
        self,
        ffmpeg_writer_cls,
        gstreamer_writer_cls,
    ):
        gst_writer = _DummyWriter(False)
        ffmpeg_hw_writer = _DummyWriter(True)
        ffmpeg_writer_cls.return_value = ffmpeg_hw_writer
        gstreamer_writer_cls.return_value = gst_writer

        writer, meta = video_io.create_h264_video_writer("demo.mp4", 1920, 1080, 25.0)

        self.assertIs(writer, ffmpeg_hw_writer)
        self.assertEqual(meta["writer_mode"], "hw")
        self.assertEqual(meta["writer_backend"], "ffmpeg")
        self.assertEqual(meta["fallback_reason"], "gstreamer_hw_open_failed")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw", "ffmpeg_hw"])
        self.assertTrue(gst_writer.released)

    @patch("cleaningcar.video_io.GstreamerH264Writer")
    @patch("cleaningcar.video_io.FfmpegH264Writer")
    def test_create_h264_video_writer_reports_failure_after_hardware_failures(
        self,
        ffmpeg_writer_cls,
        gstreamer_writer_cls,
    ):
        ffmpeg_hw_writer = _DummyWriter(False)
        gst_writer = _DummyWriter(False)
        ffmpeg_writer_cls.return_value = ffmpeg_hw_writer
        gstreamer_writer_cls.return_value = gst_writer

        writer, meta = video_io.create_h264_video_writer("demo.mp4", 1920, 1080, 25.0)

        self.assertIsNone(writer)
        self.assertEqual(meta["writer_mode"], "none")
        self.assertEqual(meta["writer_backend"], "none")
        self.assertEqual(meta["fallback_reason"], "hardware_writer_open_failed")
        self.assertEqual(meta["attempt_order"], ["gstreamer_hw", "ffmpeg_hw"])
        self.assertTrue(gst_writer.released)
        self.assertTrue(ffmpeg_hw_writer.released)


if __name__ == "__main__":
    unittest.main()
