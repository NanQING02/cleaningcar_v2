import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_json_safe(item) for item in value]
    item_fn = getattr(value, 'item', None)
    if callable(item_fn):
        try:
            return _json_safe(item_fn())
        except Exception:
            pass
    return str(value)


def _sanitize_source(source):
    text = str(source or '').strip()
    if not text or '://' not in text:
        return text
    try:
        parsed = urlsplit(text)
    except Exception:
        return ''
    host = parsed.hostname or ''
    if parsed.port:
        host = f'{host}:{parsed.port}'
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ''))


def _safe_name(value, fallback='run'):
    text = str(value or '').strip()
    safe = ''.join(char if char.isalnum() or char in ('-', '_', '.') else '_' for char in text)
    return safe.strip('._') or fallback


class EventTraceRecorder:
    def __init__(self, enabled=False, trace_dir=None, metadata=None, queue_size=4096):
        self.enabled = bool(enabled)
        self.run_dir = None
        self._queue = None
        self._thread = None
        self._closed = False
        self._dropped_records = 0
        self._counts = Counter()
        self._event_type_counts = Counter()
        if not self.enabled:
            return
        root = Path(trace_dir or 'event_traces').expanduser()
        config_name = _safe_name((metadata or {}).get('configName'), fallback='config')
        run_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{uuid4().hex[:8]}'
        self.run_dir = root / config_name / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        run_metadata = dict(metadata or {})
        run_metadata.update({
            'runId': run_id,
            'createdAt': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pid': os.getpid(),
        })
        (self.run_dir / 'run.json').write_text(
            json.dumps(_json_safe(run_metadata), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        self._queue = Queue(maxsize=max(128, int(queue_size or 4096)))
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @classmethod
    def from_config(cls, config, fps):
        logic = (config or {}).get('logic', {}) or {}
        if not bool(logic.get('event_trace_enabled', False)):
            return cls(enabled=False)
        trace_dir = Path(str(logic.get('event_trace_dir', 'event_traces') or 'event_traces').strip())
        if not trace_dir.is_absolute():
            config_path_text = str((config or {}).get('config_path', '') or '').strip()
            base_dir = Path(config_path_text).resolve().parent.parent if config_path_text else Path.cwd()
            trace_dir = base_dir / trace_dir
        metadata = {
            'configName': (config or {}).get('config_name', ''),
            'configPath': (config or {}).get('config_path', ''),
            'deviceId': ((config or {}).get('system', {}) or {}).get('device_id', ''),
            'lane': (config or {}).get('lane_name', ''),
            'source': _sanitize_source(((config or {}).get('video', {}) or {}).get('source', '')),
            'sourceMode': ((config or {}).get('video', {}) or {}).get('source_mode', ''),
            'fps': float(fps or 0.0),
            'zones': (config or {}).get('zones', {}),
        }
        try:
            return cls(
                enabled=True,
                trace_dir=trace_dir,
                metadata=metadata,
                queue_size=logic.get('event_trace_queue_size', 4096),
            )
        except Exception as exc:
            print(f'[event-trace] disabled after init failure: {exc}')
            return cls(enabled=False)

    def record(self, kind, payload=None):
        if not self.enabled or self._closed or self._queue is None:
            return False
        record = {
            'kind': str(kind or 'unknown'),
            'recordedAt': time.time(),
            'data': _json_safe(payload or {}),
        }
        try:
            self._queue.put_nowait(record)
            return True
        except Full:
            self._dropped_records += 1
            return False

    def _worker(self):
        frames_path = self.run_dir / 'frames.jsonl'
        events_path = self.run_dir / 'events.jsonl'
        with frames_path.open('a', encoding='utf-8') as frames_file, events_path.open('a', encoding='utf-8') as events_file:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except Empty:
                    if self._closed:
                        break
                    continue
                if item is None:
                    self._queue.task_done()
                    break
                kind = str(item.get('kind') or 'unknown')
                target = events_file if kind in {'event', 'sequence_issue'} else frames_file
                target.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
                target.flush()
                self._counts[kind] += 1
                if kind == 'event':
                    event_type = ((item.get('data') or {}).get('event') or {}).get('type')
                    if event_type is not None:
                        self._event_type_counts[str(event_type)] += 1
                self._queue.task_done()

    def close(self):
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self._queue is not None:
            try:
                self._queue.put(None, timeout=1.0)
            except Full:
                self._dropped_records += 1
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        summary = {
            'closedAt': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recordCounts': dict(sorted(self._counts.items())),
            'eventTypeCounts': dict(sorted(self._event_type_counts.items())),
            'droppedRecords': int(self._dropped_records),
        }
        (self.run_dir / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
