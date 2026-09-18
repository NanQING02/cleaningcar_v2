import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from utils.upload_queue import (
    PLATFORM_EVENT_ID_MAX_LENGTH,
    SQLiteUploadQueue,
    normalize_event_id,
)


class UploadQueueOrderingTests(unittest.TestCase):
    def test_same_event_waits_for_failed_earlier_stage_while_other_event_can_continue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = SQLiteUploadQueue(Path(tmpdir) / 'queue.db')
            try:
                queue.enqueue({'id': 'event-a', 'type': 1})
                queue.enqueue({'id': 'event-a', 'type': 2})
                queue.enqueue({'id': 'event-b', 'type': 1})

                first_id, first_payload, _ = queue.next_job()
                self.assertEqual((first_payload['id'], first_payload['type']), ('event-a', 1))
                queue.mark_failure(first_id, retries=1, delay_seconds=3600)

                other_id, other_payload, _ = queue.next_job()
                self.assertEqual((other_payload['id'], other_payload['type']), ('event-b', 1))
                queue.mark_success(other_id)
                self.assertIsNone(queue.next_job())

                with queue._lock:
                    queue.conn.execute('UPDATE queue SET next_retry=0 WHERE id=?', (first_id,))
                    queue.conn.commit()
                retry_id, retry_payload, _ = queue.next_job()
                self.assertEqual(retry_id, first_id)
                self.assertEqual((retry_payload['id'], retry_payload['type']), ('event-a', 1))
                queue.mark_success(retry_id)

                second_id, second_payload, _ = queue.next_job()
                self.assertEqual((second_payload['id'], second_payload['type']), ('event-a', 2))
                queue.mark_success(second_id)
                self.assertIsNone(queue.next_job())
            finally:
                queue.close()

    def test_payload_without_event_id_is_not_group_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = SQLiteUploadQueue(Path(tmpdir) / 'queue.db')
            try:
                queue.enqueue({'photoUrl': '/tmp/first.jpg', 'type': 'left'})
                queue.enqueue({'photoUrl': '/tmp/second.jpg', 'type': 'right'})

                first_id, _, _ = queue.next_job()
                queue.mark_failure(first_id, retries=1, delay_seconds=3600)
                second_id, second_payload, _ = queue.next_job()

                self.assertNotEqual(first_id, second_id)
                self.assertEqual(second_payload['photoUrl'], '/tmp/second.jpg')
            finally:
                queue.close()

    def test_long_event_id_is_stably_shortened_with_hash_suffix(self):
        raw = 'RK3588-VERY-LONG-DEVICE-NAME-' + 'x' * 80
        other = raw + '-different'

        normalized = normalize_event_id(raw)
        repeated = normalize_event_id(raw)
        normalized_other = normalize_event_id(other)

        self.assertEqual(normalized, repeated)
        self.assertLessEqual(len(normalized), PLATFORM_EVENT_ID_MAX_LENGTH)
        self.assertLessEqual(len(normalized_other), PLATFORM_EVENT_ID_MAX_LENGTH)
        self.assertNotEqual(normalized, normalized_other)

    def test_existing_queue_schema_is_migrated_and_legacy_long_ids_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'queue.db'
            raw_id = 'legacy-event-' + 'y' * 80
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                'CREATE TABLE queue ('
                'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'payload TEXT NOT NULL,'
                'retries INTEGER NOT NULL DEFAULT 0,'
                'next_retry REAL NOT NULL DEFAULT 0,'
                'created REAL NOT NULL'
                ')'
            )
            for event_type in (1, 2):
                conn.execute(
                    'INSERT INTO queue (payload, retries, next_retry, created) VALUES (?, 0, 0, 0)',
                    (json.dumps({'id': raw_id, 'type': event_type}, ensure_ascii=False),),
                )
            conn.commit()
            conn.close()

            queue = SQLiteUploadQueue(db_path)
            try:
                columns = {
                    row[1]
                    for row in queue.conn.execute('PRAGMA table_info(queue)').fetchall()
                }
                rows = queue.conn.execute(
                    'SELECT payload, group_key, event_type FROM queue ORDER BY id'
                ).fetchall()

                self.assertIn('group_key', columns)
                self.assertIn('event_type', columns)
                self.assertEqual(rows[0][1], rows[1][1])
                self.assertLessEqual(len(rows[0][1]), PLATFORM_EVENT_ID_MAX_LENGTH)
                self.assertEqual([row[2] for row in rows], [1, 2])
                self.assertEqual(json.loads(rows[0][0])['id'], rows[0][1])
            finally:
                queue.close()


if __name__ == '__main__':
    unittest.main()
