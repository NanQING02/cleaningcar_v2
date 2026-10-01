import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    @staticmethod
    def _make_finalized_writer(tmpdir, frames=250, fps=25.0, first_ts=100.0, last_ts=149.96):
        root = Path(tmpdir)
        writer = video_io.FfmpegH264Writer.__new__(video_io.FfmpegH264Writer)
        writer.path = str(root / "event.mp4")
        writer._output_path = str(root / "event_temp.mp4")
        Path(writer._output_path).write_bytes(b"encoded-video")
        writer._frames_total = frames
        writer.fps = fps
        writer._first_capture_ts = first_ts
        writer._last_capture_ts = last_ts
        writer.finalized_duration = 0.0
        return writer

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

    @patch("cleaningcar.video_io.subprocess.run")
    def test_retime_finalized_file_uses_capture_timestamps_without_reencoding(self, run):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_finalized_writer(tmpdir)

            def _run_media_tool(cmd, **_kwargs):
                if cmd[0] == 'ffprobe':
                    return SimpleNamespace(returncode=0, stdout='50.040000\n')
                Path(cmd[-1]).write_bytes(b"retimed-video")
                return SimpleNamespace(returncode=0, stdout='')

            run.side_effect = _run_media_tool

            result = writer._retime_finalized_file()

            self.assertTrue(result)
            self.assertAlmostEqual(writer.finalized_duration, 50.04, places=2)
            self.assertEqual(Path(writer.path).read_bytes(), b"retimed-video")
            self.assertFalse(Path(writer._output_path).exists())
            cmd = run.call_args_list[0].args[0]
            self.assertIn("-itsscale", cmd)
            self.assertAlmostEqual(float(cmd[cmd.index("-itsscale") + 1]), 5.0, places=6)
            self.assertEqual(cmd[cmd.index("-c") + 1], "copy")

    @patch("cleaningcar.video_io.subprocess.run", side_effect=FileNotFoundError("ffmpeg"))
    def test_retime_failure_preserves_original_playable_file(self, _run):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_finalized_writer(tmpdir)

            result = writer._retime_finalized_file()

            self.assertTrue(result)
            self.assertAlmostEqual(writer.finalized_duration, 10.0, places=2)
            self.assertEqual(Path(writer.path).read_bytes(), b"encoded-video")
            self.assertFalse(Path(writer._output_path).exists())

    @patch("cleaningcar.video_io.subprocess.run")
    def test_probe_duration_falls_back_when_ffprobe_output_is_invalid(self, run):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_finalized_writer(tmpdir)
            Path(writer.path).write_bytes(b"video")
            run.return_value = SimpleNamespace(returncode=0, stdout='N/A\n')

            duration = writer._probe_finalized_duration(12.5)

            self.assertEqual(duration, 12.5)


if __name__ == "__main__":
    unittest.main()
