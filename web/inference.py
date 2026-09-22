import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from fastapi import HTTPException

from cleaningcar.runtime_signals import load_json_file, resolve_runtime_settings, write_json_atomic
from config_manager import ConfigError, ConfigManager

from . import state


ROUTINE_LOG_INTERVAL_SECONDS = {
    'perf': 60.0,
    'diag': 60.0,
    'monitor': 60.0,
    'npu_status': 60.0,
    'per_id_video': 60.0,
    'rknn_static': 3600.0,
}


def _safe_log_component(value, fallback):
    text = str(value or '').strip()
    cleaned = ''.join(char if (char.isalnum() or char in {'-', '_'}) else '-' for char in text)
    cleaned = cleaned.strip('-_')
    return (cleaned or fallback)[:48]


def _inference_routine_category(line):
    text = str(line or '').strip()
    if text.startswith('[perf]'):
        return 'perf'
    if text.startswith('[diag]'):
        return 'diag'
    if text.startswith('[monitor]'):
        return 'monitor'
    if text.startswith('[npu-status]'):
        return 'npu_status'
    if text.startswith('[per-id-video] async') or text.startswith('[per-id-video] encoder='):
        return 'per_id_video'
    if (
        'RKNN_QUERY_INPUT_DYNAMIC_RANGE' in text
        or 'Query dynamic range failed' in text
        or text.startswith('W rknn-toolkit-lite2 version:')
    ):
        return 'rknn_static'
    return ''


def _build_inference_log_name(config_path, device_id, launch_id, started_at):
    config_key = _safe_log_component(Path(config_path).stem, 'config')
    device_key = _safe_log_component(device_id, 'device')
    launch_key = _safe_log_component(launch_id, 'launch')
    stamp = datetime.fromtimestamp(float(started_at)).strftime('%Y%m%d_%H%M%S')
    return f'infer_{config_key}_{device_key}_{stamp}_{launch_key}.log'


class InferenceManager:
    def __init__(self, script_path: Path, config_path: Path):
        self.script_path = Path(script_path)
        self.config_path = Path(config_path)
        self.process: Optional[subprocess.Popen] = None
        self.file_source = False
        self.single_shot = False
        self.lock = threading.Lock()
        self.desired = False
        self.auto_restart = True
        self.auto_restart_user_set = False
        self.restart_count = 0
        self.last_start: Optional[float] = None
        self.last_exit: Optional[Dict[str, float]] = None
        self.heartbeat_path: Optional[Path] = None
        self.startup_flag_path: Optional[Path] = None
        self.command_dir: Optional[Path] = None
        self.device_id: str = ''
        self.runtime_namespace_key: str = ''
        self.launch_id: str = ''
        self.heartbeat_timeout_seconds = 30.0
        self.progress_timeout_seconds = 90.0
        self.heartbeat_startup_grace_seconds = 90.0
        self.auto_restart_max_attempts = 3
        self.auto_restart_window_seconds = 600.0
        self.auto_restart_backoff_seconds = 5.0
        self.auto_restart_backoff_max_seconds = 60.0
        self.automatic_restart_times = deque(maxlen=64)
        self.restart_history = deque(maxlen=32)
        self.auto_restart_suspended = False
        self.next_restart_at: Optional[float] = None
        self.last_restart_reason = ''
        self.last_heartbeat_status: Dict[str, object] = {
            'available': False,
            'healthy': True,
            'reason': 'not_started',
        }
        self._wash_priority_bypass_paused = False
        self._wash_priority_resume_due: Optional[float] = None
        self._wash_priority_last_target = ''
        self._wash_priority_last_target_config: Optional[Path] = None
        self._wash_priority_last_missing_log = ''
        self.log_buffer = deque(maxlen=800)
        self.log_lock = threading.Lock()
        self._routine_last_emit: Dict[str, float] = {}
        self._routine_suppressed: Dict[str, int] = {}
        self.shutdown = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _append_log(self, message: str):
        ts = time.time()
        stamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        line = f'{stamp} {message}'
        print(line)
        with self.log_lock:
            self.log_buffer.append((ts, message))

    def _write_captured_line(self, line, log_file, log_label, now=None):
        now = time.time() if now is None else float(now)
        category = _inference_routine_category(line)
        suppressed = 0
        if category:
            interval = ROUTINE_LOG_INTERVAL_SECONDS.get(category, 60.0)
            last_emit = self._routine_last_emit.get(category)
            if last_emit is not None and now - last_emit < interval:
                self._routine_suppressed[category] = self._routine_suppressed.get(category, 0) + 1
                return False
            self._routine_last_emit[category] = now
            suppressed = self._routine_suppressed.pop(category, 0)
        if suppressed:
            summary = f'[log-throttle] {category} 已省略 {suppressed} 条重复例行日志'
            if log_file is not None:
                log_file.write(summary + '\n')
            self._append_log(f'[infer:{log_label}] {summary}')
        if log_file is not None:
            log_file.write(line + '\n')
        self._append_log(f'[infer:{log_label}] {line}')
        return True

    def _flush_suppressed_log_summaries(self, log_file, log_label):
        for category, count in sorted(self._routine_suppressed.items()):
            if count <= 0:
                continue
            summary = f'[log-throttle] {category} 进程结束前共省略 {count} 条重复例行日志'
            if log_file is not None:
                log_file.write(summary + '\n')
            self._append_log(f'[infer:{log_label}] {summary}')
        self._routine_suppressed.clear()

    def _capture_output(self, proc: subprocess.Popen, log_path: Optional[Path] = None, log_label='inference'):
        if not proc.stdout:
            return
        f = None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                f = log_path.open("a", encoding="utf-8")
            except Exception:
                f = None
        try:
            for raw in proc.stdout:
                if not raw:
                    break
                line = raw.rstrip()
                try:
                    self._write_captured_line(line, f, log_label)
                except Exception:
                    self._append_log(f'[infer:{log_label}] {line}')
        finally:
            try:
                self._flush_suppressed_log_summaries(f, log_label)
            except Exception:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

    @staticmethod
    def _safe_getpgid(pid: int) -> Optional[int]:
        if pid <= 0:
            return None
        try:
            return os.getpgid(pid)
        except OSError:
            return None

    def _signal_target(self, pid: int, sig: int, reason: str, allow_group: bool = True) -> bool:
        if pid <= 0:
            return False
        pgid = self._safe_getpgid(pid) if allow_group else None
        if allow_group and pgid is not None and pgid == pid:
            try:
                os.killpg(pgid, sig)
                return True
            except OSError as exc:
                self._append_log(
                    f'[guardian] failed to signal process group pgid={pgid} sig={sig} reason={reason}: {exc}'
                )
                return False
        try:
            os.kill(pid, sig)
            return True
        except OSError as exc:
            self._append_log(f'[guardian] failed to signal pid={pid} sig={sig} reason={reason}: {exc}')
            return False

    def _launch_locked(self):
        if not self.script_path.exists():
            raise RuntimeError('run_zone_detect.py not found')
        self._load_watchdog_settings_locked()
        self._terminate_stale_runtime_process_locked()
        self._clear_heartbeat_file_locked()
        self._sync_source_policy_locked()
        if self.file_source:
            if self.auto_restart:
                self._append_log('[guardian] file source detected，自动重启已开启，跑完整个文件后会重新开始')
            else:
                self._append_log('[guardian] file source detected，自动重启默认关闭，本次推理完成后将停止')
        cmd = [sys.executable, str(self.script_path), "--config", str(self.config_path)]
        self.launch_id = uuid4().hex[:12]
        env = os.environ.copy()
        env['CLEANINGCAR_LAUNCH_ID'] = self.launch_id
        env['CLEANINGCAR_RUNTIME_NAMESPACE_KEY'] = self.runtime_namespace_key
        env['CLEANINGCAR_DEVICE_ID'] = self.device_id
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self.process = proc
        self.restart_count += 1
        self.last_start = time.time()
        self._routine_last_emit.clear()
        self._routine_suppressed.clear()
        self.last_heartbeat_status = {
            'available': False,
            'healthy': True,
            'reason': 'starting',
            'path': str(self.heartbeat_path) if self.heartbeat_path else '',
            'runtime_namespace_key': self.runtime_namespace_key,
            'launch_id': self.launch_id,
        }
        log_dir = state.ROOT / "logs" / "inference"
        log_name = _build_inference_log_name(
            self.config_path,
            self.device_id,
            self.launch_id,
            self.last_start,
        )
        log_path = log_dir / log_name
        log_label = _safe_log_component(self.config_path.stem, 'inference')
        threading.Thread(
            target=self._capture_output,
            args=(proc, log_path, log_label),
            daemon=True,
        ).start()
        self._append_log(
            f'[guardian] started inference pid={proc.pid} '
            f'ns={self.runtime_namespace_key} launch={self.launch_id} log={log_path}'
        )

    def _terminate_locked(self):
        if not self.process:
            return
        proc = self.process
        pgid = self._safe_getpgid(proc.pid)
        if pgid is not None and pgid == proc.pid:
            self._append_log(f'[guardian] stopping pid={proc.pid} pgid={pgid}')
        else:
            self._append_log(f'[guardian] stopping pid={proc.pid}')
        try:
            self._signal_target(proc.pid, signal.SIGTERM, reason='tracked_stop', allow_group=True)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._signal_target(proc.pid, getattr(signal, 'SIGKILL', signal.SIGTERM), reason='tracked_kill', allow_group=True)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        finally:
            code = proc.poll()
            self.last_exit = {'time': time.time(), 'code': code if code is not None else -1}
            self.process = None

    def start(self):
        with self.lock:
            self._sync_source_policy_locked()
            self._reset_restart_guard_locked()
            self.desired = True
            if not self.process or self.process.poll() is not None:
                self._launch_locked()
        return self.status()

    def stop(self):
        with self.lock:
            self.desired = False
            self.next_restart_at = None
            self._terminate_locked()
        return self.status()

    def restart(self):
        with self.lock:
            self._sync_source_policy_locked()
            self._reset_restart_guard_locked()
            self.desired = True
            self._terminate_locked()
            self._launch_locked()
        return self.status()

    def set_auto_restart(self, enabled: bool):
        with self.lock:
            self.auto_restart = bool(enabled)
            self.auto_restart_user_set = True
            self.single_shot = bool(self.file_source and not self.auto_restart)
            if self.auto_restart:
                self._reset_restart_guard_locked()
            else:
                self.next_restart_at = None
        self._append_log(f'[guardian] auto_restart set to {enabled}')

    def status(self):
        with self.lock:
            self._sync_source_policy_locked()
            running = bool(self.process and self.process.poll() is None)
            pid = self.process.pid if running else None
            last_start = self.last_start
            last_exit = self.last_exit
            restart_count = self.restart_count
            auto_restart = self.auto_restart
            heartbeat = dict(self.last_heartbeat_status)
            file_source = self.file_source
            single_shot = self.single_shot
            runtime_namespace_key = self.runtime_namespace_key
            device_id = self.device_id
            launch_id = self.launch_id
            heartbeat_path = str(self.heartbeat_path) if self.heartbeat_path else ''
            auto_restart_suspended = self.auto_restart_suspended
            next_restart_at = self.next_restart_at
            last_restart_reason = self.last_restart_reason
            restart_history = list(self.restart_history)
        status = {
            'running': running,
            'pid': pid,
            'last_start': last_start,
            'last_exit': last_exit,
            'restart_count': restart_count,
            'auto_restart': auto_restart,
            'file_source': file_source,
            'single_shot': single_shot,
            'heartbeat': heartbeat,
            'runtime_namespace_key': runtime_namespace_key,
            'device_id': device_id,
            'launch_id': launch_id,
            'heartbeat_path': heartbeat_path,
            'auto_restart_suspended': auto_restart_suspended,
            'next_restart_at': next_restart_at,
            'last_restart_reason': last_restart_reason,
            'restart_history': restart_history,
        }
        return status

    def logs(self, limit: int = 200):
        limit = max(1, min(1000, int(limit)))
        with self.log_lock:
            items = list(self.log_buffer)[-limit:]
        result = []
        for ts, line in items:
            result.append({'timestamp': datetime.fromtimestamp(ts).isoformat(timespec='seconds'), 'line': line})
        return result

    def _detect_file_source_locked(self, log_error: bool = False) -> bool:
        try:
            cfg = ConfigManager(self.config_path)
        except ConfigError as exc:
            if log_error:
                self._append_log(f'[guardian] config error: {exc}')
            return False
        video_cfg = cfg.video
        source = str(video_cfg.get('source', '')).strip()
        if not source:
            return False
        lowered = source.lower()
        if lowered.startswith(('rtsp://', 'rtmp://', 'rtp://', 'rtsps://', 'http://', 'https://')):
            return False
        candidate = Path(source).expanduser()
        if not candidate.is_absolute():
            candidate = (self.config_path.parent / candidate).resolve()
        if candidate.is_file():
            return True
        mode = str(video_cfg.get('source_mode', '')).lower()
        if mode == 'camera':
            return False
        return mode == 'file'

    def _sync_source_policy_locked(self):
        self.file_source = self._detect_file_source_locked()
        if not self.auto_restart_user_set:
            self.auto_restart = not self.file_source
        self.single_shot = bool(self.file_source and not self.auto_restart)

    def _reset_restart_guard_locked(self):
        self.automatic_restart_times.clear()
        self.restart_history.clear()
        self.auto_restart_suspended = False
        self.next_restart_at = None
        self.last_restart_reason = ''

    def _schedule_auto_restart_locked(self, reason, now=None):
        now = time.time() if now is None else float(now)
        reason = str(reason or 'unknown')
        cutoff = now - max(1.0, float(self.auto_restart_window_seconds))
        while self.automatic_restart_times and self.automatic_restart_times[0] < cutoff:
            self.automatic_restart_times.popleft()
        max_attempts = max(0, int(self.auto_restart_max_attempts))
        if max_attempts and len(self.automatic_restart_times) >= max_attempts:
            self.auto_restart_suspended = True
            self.next_restart_at = None
            self.last_restart_reason = reason
            self.restart_history.append({
                'time': now,
                'reason': reason,
                'action': 'suspended',
            })
            self._append_log(
                f'[guardian] auto restart suspended after {len(self.automatic_restart_times)} '
                f'attempts in {self.auto_restart_window_seconds:.0f}s reason={reason}'
            )
            return False
        attempt_index = len(self.automatic_restart_times)
        delay = min(
            max(0.0, float(self.auto_restart_backoff_seconds)) * (2 ** attempt_index),
            max(0.0, float(self.auto_restart_backoff_max_seconds)),
        )
        self.automatic_restart_times.append(now)
        self.next_restart_at = now + delay
        self.last_restart_reason = reason
        self.restart_history.append({
            'time': now,
            'reason': reason,
            'action': 'scheduled',
            'delay_seconds': delay,
            'attempt': len(self.automatic_restart_times),
        })
        self._append_log(
            f'[guardian] auto restart scheduled attempt={len(self.automatic_restart_times)} '
            f'delay={delay:.1f}s reason={reason}'
        )
        return True

    def _process_exit_reason_locked(self, code):
        reason = f'process_exit:{code}'
        if not self.heartbeat_path:
            return reason
        heartbeat = load_json_file(self.heartbeat_path) or {}
        runtime_status = str(heartbeat.get('status') or '').strip().lower()
        failure_reason = str(heartbeat.get('failure_reason') or '').strip()
        if runtime_status.endswith('failed'):
            reason = runtime_status
            if failure_reason:
                reason = f'{reason}:{failure_reason}'
        return reason

    @staticmethod
    def _coerce_float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value) -> Optional[int]:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            return None
        return pid if pid > 0 else None

    @staticmethod
    def _pid_exists(pid: Optional[int]) -> bool:
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _terminate_pid_locked(self, pid: int, reason: str) -> bool:
        tracked_pid = self.process.pid if self.process and self.process.poll() is None else None
        if pid <= 0 or pid == tracked_pid or pid == os.getpid():
            return False
        if not self._pid_exists(pid):
            return False
        pgid = self._safe_getpgid(pid)
        if pgid is not None and pgid == pid:
            self._append_log(f'[guardian] terminating stale inference pid={pid} pgid={pgid} reason={reason}')
        else:
            self._append_log(f'[guardian] terminating stale inference pid={pid} reason={reason}')
        if not self._signal_target(pid, signal.SIGTERM, reason=f'stale_term:{reason}', allow_group=True):
            return False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._pid_exists(pid):
                self._append_log(f'[guardian] stale inference exited pid={pid}')
                return True
            time.sleep(0.1)
        force_signal = getattr(signal, 'SIGKILL', signal.SIGTERM)
        if not self._signal_target(pid, force_signal, reason=f'stale_kill:{reason}', allow_group=True):
            return False
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if not self._pid_exists(pid):
                self._append_log(f'[guardian] stale inference killed pid={pid}')
                return True
            time.sleep(0.1)
        self._append_log(f'[guardian] stale inference still alive pid={pid}')
        return False

    def _load_watchdog_settings_locked(self):
        try:
            cfg = ConfigManager(self.config_path)
            runtime = resolve_runtime_settings(cfg.data, self.config_path.parent, namespace_hint=self.config_path.stem)
            system_cfg = cfg.system
        except ConfigError as exc:
            self._append_log(f'[guardian] config error: {exc}')
            runtime = resolve_runtime_settings({}, self.config_path.parent, namespace_hint=self.config_path.stem)
            system_cfg = {}
        self.heartbeat_path = Path(runtime['heartbeat_path']) if runtime.get('heartbeat_path') else None
        self.startup_flag_path = Path(runtime['startup_flag_path']) if runtime.get('startup_flag_path') else None
        self.command_dir = Path(runtime['command_dir']) if runtime.get('command_dir') else None
        self.runtime_namespace_key = str(runtime.get('runtime_namespace_key') or '')
        self.device_id = str(runtime.get('device_id') or '')
        self.heartbeat_timeout_seconds = float(runtime.get('heartbeat_timeout_seconds', 30.0) or 30.0)
        self.progress_timeout_seconds = float(runtime.get('progress_timeout_seconds', 90.0) or 90.0)
        self.heartbeat_startup_grace_seconds = float(
            runtime.get('heartbeat_startup_grace_seconds', self.progress_timeout_seconds) or self.progress_timeout_seconds
        )
        self.auto_restart_max_attempts = max(
            0,
            int(system_cfg.get('auto_restart_max_attempts', 3) or 0),
        )
        self.auto_restart_window_seconds = max(
            1.0,
            float(system_cfg.get('auto_restart_window_seconds', 600.0) or 600.0),
        )
        self.auto_restart_backoff_seconds = max(
            0.0,
            float(system_cfg.get('auto_restart_backoff_seconds', 5.0) or 0.0),
        )
        self.auto_restart_backoff_max_seconds = max(
            self.auto_restart_backoff_seconds,
            float(system_cfg.get('auto_restart_backoff_max_seconds', 60.0) or 60.0),
        )

    def _terminate_stale_runtime_process_locked(self):
        if not self.heartbeat_path:
            return
        heartbeat = load_json_file(self.heartbeat_path) or {}
        heartbeat_pid = self._coerce_int(heartbeat.get('pid'))
        if heartbeat_pid is None:
            return
        heartbeat_namespace_key = str(heartbeat.get('runtime_namespace_key') or '').strip()
        if (
            heartbeat_namespace_key
            and self.runtime_namespace_key
            and heartbeat_namespace_key != self.runtime_namespace_key
        ):
            self._append_log(
                f'[guardian] shared heartbeat path has different namespace, skip stale cleanup: '
                f'path={self.heartbeat_path} file_ns={heartbeat_namespace_key} current_ns={self.runtime_namespace_key}'
            )
            return
        self._terminate_pid_locked(heartbeat_pid, f'runtime_namespace={self.runtime_namespace_key or "default"}')

    def _clear_heartbeat_file_locked(self):
        if not self.heartbeat_path:
            return
        try:
            if self.heartbeat_path.exists():
                self.heartbeat_path.unlink()
                self._append_log(f'[guardian] cleared stale heartbeat file: {self.heartbeat_path}')
        except Exception as exc:
            self._append_log(f'[guardian] failed to clear heartbeat file {self.heartbeat_path}: {exc}')

    def _check_heartbeat_locked(self, now: Optional[float] = None) -> Dict[str, object]:
        now = time.time() if now is None else float(now)
        info: Dict[str, object] = {
            'available': False,
            'healthy': True,
            'reason': 'disabled',
            'path': str(self.heartbeat_path) if self.heartbeat_path else '',
            'runtime_namespace_key': self.runtime_namespace_key,
            'launch_id': self.launch_id,
            'heartbeat_timeout_seconds': self.heartbeat_timeout_seconds,
            'progress_timeout_seconds': self.progress_timeout_seconds,
            'startup_grace_seconds': self.heartbeat_startup_grace_seconds,
        }
        if not self.heartbeat_path:
            return info
        if not self.last_start:
            info['reason'] = 'not_started'
            return info

        startup_age = max(0.0, now - self.last_start)
        startup_in_grace = startup_age <= self.heartbeat_startup_grace_seconds
        info['startup_age_seconds'] = startup_age
        if not self.heartbeat_path.exists():
            info['reason'] = 'heartbeat_missing'
            info['healthy'] = startup_in_grace
            return info

        info['available'] = True
        try:
            stat = self.heartbeat_path.stat()
            info['file_mtime'] = stat.st_mtime
        except OSError:
            stat = None
            info['file_mtime'] = None

        heartbeat = load_json_file(self.heartbeat_path) or {}
        info['heartbeat'] = heartbeat
        runtime_status = str(heartbeat.get('status') or '').strip().lower()
        info['runtime_status'] = runtime_status

        heartbeat_ts = self._coerce_float(heartbeat.get('timestamp'))
        if heartbeat_ts is None and stat is not None:
            heartbeat_ts = stat.st_mtime
        heartbeat_pid = heartbeat.get('pid')
        heartbeat_launch_id = str(heartbeat.get('launch_id') or '').strip()
        expected_pid = self.process.pid if self.process and self.process.poll() is None else None
        old_heartbeat = False
        mismatch_reason = ''
        if heartbeat_ts is not None and heartbeat_ts + 1e-6 < self.last_start:
            old_heartbeat = True
            mismatch_reason = 'heartbeat_old'
        if expected_pid is not None and heartbeat_pid not in (None, '') and str(heartbeat_pid) != str(expected_pid):
            old_heartbeat = True
            info['expected_pid'] = expected_pid
            info['heartbeat_pid'] = heartbeat_pid
            mismatch_reason = 'heartbeat_pid_mismatch'
        if self.launch_id:
            info['expected_launch_id'] = self.launch_id
            info['heartbeat_launch_id'] = heartbeat_launch_id
            if heartbeat_launch_id != self.launch_id:
                old_heartbeat = True
                mismatch_reason = 'heartbeat_launch_mismatch' if heartbeat_launch_id else 'heartbeat_launch_missing'
        if old_heartbeat:
            info['old_heartbeat_detected'] = True
            if startup_in_grace:
                info['reason'] = 'starting'
                return info
            info['healthy'] = False
            info['reason'] = mismatch_reason or 'heartbeat_pid_mismatch'
            return info

        if heartbeat_ts is not None:
            heartbeat_age = max(0.0, now - heartbeat_ts)
            info['heartbeat_age_seconds'] = heartbeat_age
            if heartbeat_age > self.heartbeat_timeout_seconds:
                if startup_in_grace:
                    info['reason'] = 'starting'
                    return info
                info['healthy'] = False
                info['reason'] = 'heartbeat_stale'
                return info
        elif not startup_in_grace:
            info['healthy'] = False
            info['reason'] = 'heartbeat_timestamp_missing'
            return info

        last_progress_ts = self._coerce_float(heartbeat.get('last_progress_ts'))
        if last_progress_ts is not None:
            progress_age = max(0.0, now - last_progress_ts)
            info['progress_age_seconds'] = progress_age
            if not startup_in_grace and progress_age > self.progress_timeout_seconds:
                if runtime_status in {'waiting_reader', 'reader_reconnect', 'reader_reopen'}:
                    info['degraded'] = True
                    info['reason'] = runtime_status
                    return info
                info['healthy'] = False
                info['reason'] = 'progress_stale'
                return info
        elif not startup_in_grace:
            info['healthy'] = False
            info['reason'] = 'progress_missing'
            return info

        info['reason'] = 'ok'
        return info

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}
        return bool(value)

    def _load_wash_priority_settings_locked(self) -> Dict[str, object]:
        try:
            cfg = ConfigManager(self.config_path)
        except ConfigError as exc:
            self._append_log(f'[wash-priority] config error: {exc}')
            return {'enabled': False}
        wheel_cfg = (cfg.data or {}).get('wheel', {}) or {}
        target_key = str(wheel_cfg.get('pause_bypass_config_key', 'config_绕行.json') or 'config_绕行.json').strip()
        try:
            resume_delay = float(wheel_cfg.get('pause_bypass_resume_delay_seconds', 0.5))
        except (TypeError, ValueError):
            resume_delay = 0.5
        return {
            'enabled': self._coerce_bool(wheel_cfg.get('pause_bypass_during_wash_enabled', False)),
            'target_key': target_key,
            'resume_delay': max(0.0, resume_delay),
        }

    def _target_config_path_for_key_locked(self, key: str) -> Optional[Path]:
        key = str(key or '').strip()
        if not key:
            return None
        base = self.config_path.parent
        candidate = (base / key).resolve()
        try:
            base_resolved = base.resolve()
            candidate.relative_to(base_resolved)
        except Exception:
            return None
        if candidate.suffix.lower() != '.json' or not candidate.exists():
            return None
        return candidate

    def _write_runtime_pause_command_locked(self, target_config: Path, paused: bool, reason: str) -> bool:
        try:
            cfg = ConfigManager(target_config)
            runtime = resolve_runtime_settings(cfg.data, target_config.parent, namespace_hint=target_config.stem)
            command_dir = Path(runtime['command_dir'])
            command_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            command_path = command_dir / f"runtime_pause_{ts}_{uuid4().hex[:8]}.cmd.json"
            write_json_atomic(
                command_path,
                {
                    'cmd': 'set_runtime_pause',
                    'paused': bool(paused),
                    'reason': reason,
                    'requested_at': time.time(),
                    'requested_by': 'web_guardian',
                    'source_config': self.config_path.name,
                    'target_config': target_config.name,
                },
            )
            self._append_log(
                f'[wash-priority] sent {"pause" if paused else "resume"} command '
                f'target={target_config.name} cmd={command_path}'
            )
            return True
        except Exception as exc:
            self._append_log(f'[wash-priority] failed to write runtime pause command target={target_config}: {exc}')
            return False

    def _coordinate_wash_priority_locked(self, heartbeat_info: Dict[str, object], now: Optional[float] = None) -> None:
        now = time.time() if now is None else float(now)
        settings = self._load_wash_priority_settings_locked()
        if not settings.get('enabled'):
            self._wash_priority_resume_due = None
            if self._wash_priority_bypass_paused and self._wash_priority_last_target_config is not None:
                if not self._write_runtime_pause_command_locked(
                    self._wash_priority_last_target_config,
                    False,
                    'wash_lane_priority_disabled',
                ):
                    return
                self._wash_priority_bypass_paused = False
            self._wash_priority_last_target = ''
            self._wash_priority_last_target_config = None
            return
        target_key = str(settings.get('target_key') or '').strip()
        target_config = self._target_config_path_for_key_locked(target_key)
        if target_config is None:
            log_key = target_key or '<empty>'
            if self._wash_priority_last_missing_log != log_key:
                self._append_log(f'[wash-priority] bypass config not found or invalid: {target_key}')
                self._wash_priority_last_missing_log = log_key
            return
        heartbeat = heartbeat_info.get('heartbeat') if isinstance(heartbeat_info, dict) else None
        heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
        active = bool(heartbeat.get('wash_priority_active', False))
        target_name = target_config.name
        if self._wash_priority_last_target and self._wash_priority_last_target != target_name:
            if self._wash_priority_bypass_paused and self._wash_priority_last_target_config is not None:
                if not self._write_runtime_pause_command_locked(
                    self._wash_priority_last_target_config,
                    False,
                    'wash_lane_priority_target_changed',
                ):
                    return
                self._wash_priority_bypass_paused = False
            self._wash_priority_resume_due = None
        self._wash_priority_last_target = target_name
        self._wash_priority_last_target_config = target_config
        if active:
            self._wash_priority_resume_due = None
            if not self._wash_priority_bypass_paused:
                if self._write_runtime_pause_command_locked(target_config, True, 'wash_lane_priority'):
                    self._wash_priority_bypass_paused = True
            return
        if not self._wash_priority_bypass_paused:
            self._wash_priority_resume_due = None
            return
        if self._wash_priority_resume_due is None:
            self._wash_priority_resume_due = now + float(settings.get('resume_delay', 0.5) or 0.0)
            return
        if now >= self._wash_priority_resume_due:
            if self._write_runtime_pause_command_locked(target_config, False, 'wash_lane_priority'):
                self._wash_priority_bypass_paused = False
                self._wash_priority_resume_due = None

    def _monitor_loop(self):
        while not self.shutdown.is_set():
            now = time.time()
            with self.lock:
                self._sync_source_policy_locked()
                desired = self.desired
                auto_restart = self.auto_restart
                file_source = self.file_source
            if desired:
                should_launch = False
                with self.lock:
                    if not self.process:
                        should_launch = bool(
                            auto_restart
                            and not self.auto_restart_suspended
                            and self.next_restart_at is not None
                            and now >= self.next_restart_at
                        )
                    else:
                        code = self.process.poll()
                        if code is not None:
                            self._append_log(f'[guardian] process exited with code {code}')
                            self.last_exit = {'time': time.time(), 'code': code}
                            self.process = None
                            if file_source and not auto_restart:
                                should_launch = False
                                self.desired = False
                                self._append_log('[guardian] 本地文件源已跑完一遍，自动重启未开启，等待手动启动')
                                self.single_shot = False
                            elif auto_restart:
                                self._schedule_auto_restart_locked(
                                    self._process_exit_reason_locked(code),
                                    now=now,
                                )
                        else:
                            heartbeat = self._check_heartbeat_locked()
                            self.last_heartbeat_status = heartbeat
                            self._coordinate_wash_priority_locked(heartbeat)
                            if not heartbeat.get('healthy', True):
                                reason = heartbeat.get('reason', 'unknown')
                                self._append_log(f'[guardian] unhealthy heartbeat detected: {reason}')
                                self._terminate_locked()
                                if file_source and not auto_restart:
                                    self.desired = False
                                    self.single_shot = False
                                elif auto_restart:
                                    self._schedule_auto_restart_locked(f'heartbeat:{reason}', now=now)
                if should_launch:
                    try:
                        with self.lock:
                            self._launch_locked()
                            self.next_restart_at = None
                            self.restart_history.append({
                                'time': time.time(),
                                'reason': self.last_restart_reason,
                                'action': 'launched',
                                'pid': self.process.pid if self.process else None,
                            })
                    except Exception as exc:
                        self._append_log(f'[guardian] failed to start inference: {exc}')
                        with self.lock:
                            self.next_restart_at = None
                            if self.auto_restart and not self.auto_restart_suspended:
                                self._schedule_auto_restart_locked(
                                    f'launch_failed:{type(exc).__name__}',
                                    now=time.time(),
                                )
            else:
                with self.lock:
                    if self.process:
                        self._terminate_locked()
            time.sleep(1)

    def shutdown_manager(self):
        self.shutdown.set()
        with self.lock:
            self.desired = False
            self.auto_restart = False
            self._terminate_locked()
        self._append_log('[guardian] shutdown complete')

    def set_config_path(self, path: Path):
        with self.lock:
            self.config_path = Path(path)
            self._load_watchdog_settings_locked()
            self._sync_source_policy_locked()


INFERENCE_MANAGERS: Dict[str, InferenceManager] = {}


def _resolve_config_path_for_key(key: str) -> Path:
    key = str(key or "").strip()
    base = state.CONFIG_PATH.parent
    if not key:
        candidate = state.CONFIG_PATH
    else:
        candidate = (base / key).resolve()
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")
    if candidate.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="仅支持 JSON 配置")
    return candidate


def _get_inference_manager_for_key(key: str) -> InferenceManager:
    cfg_path = _resolve_config_path_for_key(key)
    mgr = INFERENCE_MANAGERS.get(key)
    if mgr is None:
        mgr = InferenceManager(state.RUN_SCRIPT, cfg_path)
        INFERENCE_MANAGERS[key] = mgr
    else:
        mgr.script_path = state.RUN_SCRIPT
        mgr.set_config_path(cfg_path)
    return mgr


def _get_default_inference_manager() -> InferenceManager:
    return _get_inference_manager_for_key(state.CONFIG_PATH.name)


def startup_default_manager():
    return _get_default_inference_manager()


def shutdown_all_inference_managers():
    for mgr in list(INFERENCE_MANAGERS.values()):
        try:
            mgr.shutdown_manager()
        except Exception as exc:
            print(f'[server] shutdown: failed to shutdown manager: {exc}')
