import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cleaningcar.events import EventManager, EventUploader


class _DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
        return None, {"enter_a": False, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class EventCaptureAsyncTests(unittest.TestCase):
    def _manager(self, capture_async):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        config = {
            "logic": {"event_capture_async": capture_async},
            "shadow_pool": {},
            "event_capture_dir": self._tmp.name,
            "event_output_dir": self._tmp.name,
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(64, 64),
            zone_manager=_DummyZoneManager(),
        )

    def _frame(self):
        rng = np.random.default_rng(3)
        return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_async_capture_returns_path_then_writes_file(self):
        mgr = self._manager(capture_async=True)
        frame = self._frame()

        path = mgr._save_event_capture(1, 7, 100, frame)

        self.assertTrue(path)
        self.assertTrue(mgr.wait_for_event_captures(timeout=5.0))
        self.assertTrue(Path(path).exists())
        self.assertGreater(Path(path).stat().st_size, 0)
        mgr.close()

    def test_async_capture_close_drains_queue(self):
        mgr = self._manager(capture_async=True)
        frame = self._frame()

        paths = [mgr._save_event_capture(1, 7, 100 + i, frame) for i in range(4)]

        mgr.close()
        for path in paths:
            self.assertTrue(Path(path).exists())

    def test_default_remains_synchronous(self):
        mgr = self._manager(capture_async=False)
        frame = self._frame()

        path = mgr._save_event_capture(2, 7, 200, frame)

        self.assertTrue(path)
        # 同步路径写盘即时完成
        self.assertTrue(Path(path).exists())
        self.assertIsNone(mgr._capture_worker)
        mgr.close()

    def test_async_capture_failure_on_none_frame(self):
        mgr = self._manager(capture_async=True)

        path = mgr._save_event_capture(1, 7, 100, None)

        self.assertEqual(path, '')
        mgr.close()

    def test_path_payload_is_preserved_while_async_file_is_pending(self):
        mgr = self._manager(capture_async=True)
        started = threading.Event()
        release = threading.Event()
        original = mgr._encode_event_capture

        def delayed_encode(**kwargs):
            started.set()
            release.wait(timeout=2.0)
            return original(**kwargs)

        mgr._encode_event_capture = delayed_encode
        path = mgr._save_event_capture(1, 7, 300, self._frame())
        self.assertTrue(started.wait(timeout=1.0))
        self.assertFalse(Path(path).exists())
        self.assertEqual(mgr._prepare_capture_image(path), path)
        self.assertFalse(mgr.wait_for_event_captures(timeout=0.05))
        release.set()
        self.assertTrue(mgr.wait_for_event_captures(timeout=2.0))
        mgr.close()

    def test_event_uploader_retries_until_capture_path_exists(self):
        uploader = EventUploader()
        uploader.url = 'http://127.0.0.1/event'
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = Path(tmpdir) / 'pending.jpg'
            with self.assertRaises(FileNotFoundError):
                uploader._send({'captureImage': str(capture)})

            capture.write_bytes(b'jpeg')
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = b'ok'
            response.__enter__.return_value.getcode.return_value = 200
            with patch('cleaningcar.events.urllib.request.urlopen', return_value=response) as mocked:
                result = uploader._send({'captureImage': str(capture)})
            mocked.assert_called_once()
            self.assertEqual(result['http_status'], 200)

    @staticmethod
    def _wait_for_audit_status(path, status, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
                if any(row.get('status') == status for row in rows):
                    return rows
            time.sleep(0.02)
        return []

    def test_event_uploader_audit_marks_sent_only_after_http_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_path = root / 'upload_audit.jsonl'
            with patch.object(
                EventUploader,
                '_send',
                return_value={'http_status': 201, 'response': '{"ok":true}'},
            ):
                uploader = EventUploader(
                    'http://127.0.0.1/event',
                    queue_path=root / 'queue.db',
                    audit_path=audit_path,
                )
                try:
                    result = uploader.enqueue({
                        'id': 'event-a',
                        'type': 5,
                        'captureTime': '2026-10-01 09:50:11',
                        'plateColor': '黄绿色',
                    })
                    self.assertEqual(result['status'], 'queued')
                    rows = self._wait_for_audit_status(audit_path, 'sent')
                finally:
                    uploader.close()

            self.assertEqual([row['status'] for row in rows], ['queued', 'sent'])
            sent = rows[-1]
            self.assertEqual(sent['httpStatus'], 201)
            self.assertEqual(sent['eventId'], 'event-a')
            self.assertEqual(sent['eventType'], 5)
            self.assertEqual(sent['payload']['plateColor'], '黄绿')

    def test_event_uploader_dead_letter_audit_keeps_http_error_response(self):
        error = RuntimeError('HTTP 500')
        error.code = 500
        error.read = lambda _size=4096: (
            b"MysqlDataTruncation: Data too long for column 'color' at row 1"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_path = root / 'upload_audit.jsonl'
            with patch.object(EventUploader, '_send', side_effect=error):
                uploader = EventUploader(
                    'http://127.0.0.1/event',
                    queue_path=root / 'queue.db',
                    audit_path=audit_path,
                    max_retries=1,
                )
                try:
                    uploader.enqueue({'id': 'event-b', 'type': 5, 'plateColor': '黄绿色'})
                    rows = self._wait_for_audit_status(audit_path, 'dead_letter')
                finally:
                    uploader.close()

            self.assertEqual([row['status'] for row in rows], ['queued', 'dead_letter'])
            failed = rows[-1]
            self.assertEqual(failed['httpStatus'], 500)
            self.assertIsInstance(failed['deadLetterId'], int)
            self.assertEqual(failed['payload']['plateColor'], '黄绿')
            self.assertIn('Data too long', failed['error'])
            self.assertIn('Data too long', failed['response'])

    def test_event_uploader_send_normalizes_legacy_yellow_green_payload(self):
        uploader = EventUploader()
        uploader.url = 'http://127.0.0.1/event'
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'ok'
        response.__enter__.return_value.getcode.return_value = 200

        with patch('cleaningcar.events.urllib.request.urlopen', return_value=response) as mocked:
            result = uploader._send({'id': 'legacy', 'type': 5, 'plateColor': '黄绿色'})

        request = mocked.call_args.args[0]
        body = json.loads(request.data.decode('utf-8'))
        self.assertEqual(body['plateColor'], '黄绿')
        self.assertEqual(result['payload']['plateColor'], '黄绿')


if __name__ == '__main__':
    unittest.main()
