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

    def pending(self) -> int:
        with self._lock:
            row = self.conn.execute('SELECT COUNT(1) FROM queue').fetchone()
            return int(row[0]) if row else 0

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
