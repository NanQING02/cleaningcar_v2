import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, Tuple


PLATFORM_EVENT_ID_MAX_LENGTH = 36


def normalize_event_id(value, max_length=PLATFORM_EVENT_ID_MAX_LENGTH):
    event_id = str(value or '').strip()
    max_length = max(12, int(max_length or PLATFORM_EVENT_ID_MAX_LENGTH))
    if len(event_id) <= max_length:
        return event_id
    digest = hashlib.sha1(event_id.encode('utf-8')).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    prefix = event_id[:prefix_length].rstrip('-_') or 'event'
    return f'{prefix}-{digest}'[:max_length]


class SQLiteUploadQueue:
    """Tiny persistent queue backed by SQLite for event uploads."""

    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self.conn.execute(
                'CREATE TABLE IF NOT EXISTS queue ('
                'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'payload TEXT NOT NULL,'
                'retries INTEGER NOT NULL DEFAULT 0,'
                'next_retry REAL NOT NULL DEFAULT 0,'
                'created REAL NOT NULL,'
                "group_key TEXT NOT NULL DEFAULT '',"
                'event_type INTEGER'
                ')'
            )
            columns = {
                str(row[1])
                for row in self.conn.execute('PRAGMA table_info(queue)').fetchall()
            }
            if 'group_key' not in columns:
                self.conn.execute("ALTER TABLE queue ADD COLUMN group_key TEXT NOT NULL DEFAULT ''")
            if 'event_type' not in columns:
                self.conn.execute('ALTER TABLE queue ADD COLUMN event_type INTEGER')
            self.conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_queue_group_id ON queue(group_key, id)'
            )
            self.conn.execute(
                'CREATE TABLE IF NOT EXISTS dead_letter ('
                'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'original_job_id INTEGER,'
                'payload TEXT NOT NULL,'
                'retries INTEGER NOT NULL,'
                'created REAL NOT NULL,'
                'failed_at REAL NOT NULL,'
                'last_error TEXT NOT NULL,'
                "group_key TEXT NOT NULL DEFAULT '',"
                'event_type INTEGER'
                ')'
            )
            self.conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_dead_letter_failed_at '
                'ON dead_letter(failed_at DESC)'
            )
            self.conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_dead_letter_group '
                'ON dead_letter(group_key, id)'
            )
            self._migrate_existing_rows_locked()
            if self.conn.in_transaction:
                try:
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass

    @staticmethod
    def _payload_metadata(payload):
        data = dict(payload or {})
        raw_event_id = data.get('id')
        event_id = normalize_event_id(raw_event_id) if raw_event_id not in (None, '') else ''
        if event_id:
            data['id'] = event_id
        raw_type = data.get('type')
        try:
            event_type = int(raw_type) if raw_type not in (None, '') else None
        except (TypeError, ValueError):
            event_type = None
        return data, event_id, event_type

    def _migrate_existing_rows_locked(self):
        rows = self.conn.execute(
            'SELECT id, payload, group_key, event_type FROM queue ORDER BY id'
        ).fetchall()
        for job_id, payload_json, group_key, event_type in rows:
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = {}
            normalized, resolved_group, resolved_type = self._payload_metadata(payload)
            normalized_json = json.dumps(normalized, ensure_ascii=False)
            if (
                normalized_json != payload_json
                or str(group_key or '') != resolved_group
                or event_type != resolved_type
            ):
                self.conn.execute(
                    'UPDATE queue SET payload=?, group_key=?, event_type=? WHERE id=?',
                    (normalized_json, resolved_group, resolved_type, job_id),
                )

    def enqueue(self, payload: dict):
        normalized, group_key, event_type = self._payload_metadata(payload)
        data = json.dumps(normalized, ensure_ascii=False)
        now = time.time()
        with self._lock:
            blocked = None
            if group_key:
                blocked = self.conn.execute(
                    'SELECT event_type, last_error FROM dead_letter '
                    'WHERE group_key=? ORDER BY id LIMIT 1',
                    (group_key,),
                ).fetchone()
            if blocked:
                blocker_type, blocker_error = blocked
                self.conn.execute(
                    'INSERT INTO dead_letter '
                    '(original_job_id, payload, retries, created, failed_at, last_error, group_key, event_type) '
                    'VALUES (NULL, ?, 0, ?, ?, ?, ?, ?)',
                    (
                        data,
                        now,
                        now,
                        (
                            f'blocked by existing dead-letter event type {blocker_type}: '
                            f'{str(blocker_error or "unknown error")}'
                        )[:4000],
                        group_key,
                        event_type,
                    ),
                )
                self.conn.commit()
                return
            self.conn.execute(
                'INSERT INTO queue '
                '(payload, retries, next_retry, created, group_key, event_type) '
                'VALUES (?, 0, 0, ?, ?, ?)',
                (data, now, group_key, event_type),
            )
            if self.conn.in_transaction:
                try:
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass

    def next_job(self) -> Optional[Tuple[int, dict, int]]:
        now = time.time()
        with self._lock:
            row = self.conn.execute(
                'SELECT q.id, q.payload, q.retries FROM queue AS q '
                'WHERE q.next_retry <= ? '
                'AND ('
                "q.group_key = '' OR NOT EXISTS ("
                'SELECT 1 FROM queue AS older '
                'WHERE older.group_key = q.group_key AND older.id < q.id'
                ')'
                ') '
                'AND ('
                "q.group_key = '' OR NOT EXISTS ("
                'SELECT 1 FROM dead_letter AS failed '
                'WHERE failed.group_key = q.group_key'
                ')'
                ') '
                'ORDER BY q.id LIMIT 1',
                (now,)
            ).fetchone()
            if not row:
                return None
            job_id, payload_json, retries = row
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = {}
            return job_id, payload, int(retries)

    def mark_success(self, job_id: int):
        with self._lock:
            self.conn.execute('DELETE FROM queue WHERE id=?', (job_id,))
            if self.conn.in_transaction:
                try:
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass

    def mark_failure(self, job_id: int, retries: int, delay_seconds: float):
        next_retry = time.time() + max(1.0, float(delay_seconds))
        with self._lock:
            self.conn.execute(
                'UPDATE queue SET retries=?, next_retry=? WHERE id=?',
                (retries, next_retry, job_id)
            )
            if self.conn.in_transaction:
                try:
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass

    def move_to_dead_letter(self, job_id: int, error: str, retries: Optional[int] = None) -> Optional[int]:
        error_text = str(error or 'unknown error').strip()[:4000]
        failed_at = time.time()
        with self._lock:
            row = self.conn.execute(
                'SELECT payload, retries, created, group_key, event_type '
                'FROM queue WHERE id=?',
                (job_id,),
            ).fetchone()
            if not row:
                return None
            group_key = str(row[3] or '')
            if group_key:
                rows = self.conn.execute(
                    'SELECT id, payload, retries, created, group_key, event_type '
                    'FROM queue WHERE group_key=? AND id>=? ORDER BY id',
                    (group_key, int(job_id)),
                ).fetchall()
            else:
                rows = [(int(job_id), row[0], row[1], row[2], group_key, row[4])]
            first_dead_letter_id = None
            for index, item in enumerate(rows):
                queued_id, payload, stored_retries, created, item_group, event_type = item
                item_error = error_text if index == 0 else (
                    f'blocked by earlier dead-letter event type {row[4]}: {error_text}'
                )[:4000]
                item_retries = int(retries) if index == 0 and retries is not None else int(stored_retries)
                cursor = self.conn.execute(
                    'INSERT INTO dead_letter '
                    '(original_job_id, payload, retries, created, failed_at, last_error, group_key, event_type) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        int(queued_id),
                        payload,
                        item_retries,
                        float(created),
                        failed_at,
                        item_error,
                        str(item_group or ''),
                        event_type,
                    ),
                )
                if first_dead_letter_id is None:
                    first_dead_letter_id = int(cursor.lastrowid)
            self.conn.executemany('DELETE FROM queue WHERE id=?', [(int(item[0]),) for item in rows])
            self.conn.commit()
            return first_dead_letter_id

    def dead_letters(self, limit: int = 100):
        limit = max(1, min(1000, int(limit or 100)))
        with self._lock:
            rows = self.conn.execute(
                'SELECT id, original_job_id, payload, retries, created, failed_at, '
                'last_error, group_key, event_type '
                'FROM dead_letter ORDER BY failed_at DESC, id DESC LIMIT ?',
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            try:
                payload = json.loads(row[2])
            except json.JSONDecodeError:
                payload = {}
            display_payload = dict(payload) if isinstance(payload, dict) else {}
            capture_image = display_payload.get('captureImage')
            if isinstance(capture_image, str) and len(capture_image) > 512:
                display_payload['captureImage'] = f'<omitted {len(capture_image)} characters>'
            items.append({
                'id': int(row[0]),
                'original_job_id': int(row[1]) if row[1] is not None else None,
                'payload': display_payload,
                'retries': int(row[3]),
                'created': float(row[4]),
                'failed_at': float(row[5]),
                'last_error': str(row[6] or ''),
                'group_key': str(row[7] or ''),
                'event_type': int(row[8]) if row[8] is not None else None,
            })
        return items

    def dead_letter_count(self) -> int:
        with self._lock:
            row = self.conn.execute('SELECT COUNT(1) FROM dead_letter').fetchone()
            return int(row[0]) if row else 0

    def retry_dead_letter(self, dead_letter_id: int) -> bool:
        with self._lock:
            row = self.conn.execute(
                'SELECT payload, group_key, event_type FROM dead_letter WHERE id=?',
                (int(dead_letter_id),),
            ).fetchone()
            if not row:
                return False
            payload, group_key, event_type = row
            if str(group_key or ''):
                return False
            self.conn.execute(
                'INSERT INTO queue '
                '(payload, retries, next_retry, created, group_key, event_type) '
                'VALUES (?, 0, 0, ?, ?, ?)',
                (payload, time.time(), str(group_key or ''), event_type),
            )
            self.conn.execute('DELETE FROM dead_letter WHERE id=?', (int(dead_letter_id),))
            self.conn.commit()
            return True

    def retry_dead_letter_group(self, group_key: str) -> int:
        group_key = str(group_key or '').strip()
        if not group_key:
            return 0
        with self._lock:
            rows = self.conn.execute(
                'SELECT id, payload, event_type FROM dead_letter '
                'WHERE group_key=? '
                'ORDER BY CASE WHEN event_type IS NULL THEN 1 ELSE 0 END, event_type, id',
                (group_key,),
            ).fetchall()
            for _dead_id, payload, event_type in rows:
                self.conn.execute(
                    'INSERT INTO queue '
                    '(payload, retries, next_retry, created, group_key, event_type) '
                    'VALUES (?, 0, 0, ?, ?, ?)',
                    (payload, time.time(), group_key, event_type),
                )
            if rows:
                self.conn.executemany(
                    'DELETE FROM dead_letter WHERE id=?',
                    [(int(row[0]),) for row in rows],
                )
                self.conn.commit()
            return len(rows)

    def pending(self) -> int:
        with self._lock:
            row = self.conn.execute('SELECT COUNT(1) FROM queue').fetchone()
            return int(row[0]) if row else 0

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
