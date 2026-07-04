import os
import tempfile
import time
import unittest
from pathlib import Path

from cleaningcar import video_io
from cleaningcar.storage_cleanup import RetentionPolicy, RuntimeStorageCleaner


class _DummyWriter:
    def __init__(self, path):
        self.path = str(path)
        self.release_calls = 0

    def release(self):
        self.release_calls += 1
        return True


class _DummyEventManager:
    def __init__(self):
        self.calls = []

    def emit_event(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class CleanupDisabledTests(unittest.TestCase):
    def test_runtime_storage_cleaner_keeps_expired_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            capture = root / "expired.mp4"
            capture.write_text("expired", encoding="utf-8")
            now_ts = time.time()
            os.utime(capture, (now_ts - 3 * 86400, now_ts - 3 * 86400))

            cleaner = RuntimeStorageCleaner(
                [RetentionPolicy(root=root, keep_days=1, keep_count=100, label="per-id-video")],
                interval_seconds=1,
            )

            cleaner.run_once(now=now_ts, reason="test")

            self.assertTrue(capture.exists())

    def test_runtime_storage_cleaner_keeps_files_when_count_exceeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.mp4"
            second = root / "second.mp4"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            now_ts = time.time()
            os.utime(first, (now_ts - 10, now_ts - 10))
            os.utime(second, (now_ts - 5, now_ts - 5))

            cleaner = RuntimeStorageCleaner(
                [RetentionPolicy(root=root, keep_days=0, keep_count=1, label="per-id-video")],
                interval_seconds=1,
            )

            cleaner.run_once(now=now_ts, reason="test")

            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_finalize_per_id_recording_deletes_unqualified_video_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "session.mp4"
            output_path.write_text("video", encoding="utf-8")
            writer = _DummyWriter(output_path)
            event_manager = _DummyEventManager()

            result = video_io.finalize_per_id_recording(
                writer,
                track_id=7,
                track_state={"type2_qualified": False},
                event_manager=event_manager,
            )

            self.assertFalse(result)
            self.assertFalse(output_path.exists())
            self.assertEqual(writer.release_calls, 1)
            self.assertEqual(event_manager.calls, [])


if __name__ == "__main__":
    unittest.main()
