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

from cleaningcar.runtime_signals import load_json_file, resolve_runtime_settings
from config_manager import ConfigError, ConfigManager

from . import state

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
        self.last_heartbeat_status: Dict[str, object] = {
            'available': False,
            'healthy': True,
            'reason': 'not_started',
        }
        self.log_buffer = deque(maxlen=800)
        self.log_lock = threading.Lock()
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

    def _capture_output(self, proc: subprocess.Popen, log_path: Optional[Path] = None):
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
                self._append_log(f'[infer] {line}')
                if f is not None:
                    try:
                        f.write(line + "\n")
                    except Exception:
                        pass
        finally:
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
        )
        self.process = proc
        self.restart_count += 1
        self.last_start = time.time()
        self.last_heartbeat_status = {
            'available': False,
            'healthy': True,
            'reason': 'starting',
            'path': str(self.heartbeat_path) if self.heartbeat_path else '',
            'runtime_namespace_key': self.runtime_namespace_key,
            'launch_id': self.launch_id,
        }
        log_dir = state.ROOT / "logs" / "inference"
        log_name = datetime.fromtimestamp(self.last_start).strftime("infer_%Y%m%d_%H%M%S.log")
        log_path = log_dir / log_name
        threading.Thread(target=self._capture_output, args=(proc, log_path), daemon=True).start()
        self._append_log(
            f'[guardian] started inference pid={proc.pid} '
            f'ns={self.runtime_namespace_key} launch={self.launch_id} log={log_path}'
        )

    def _terminate_locked(self):
        if not self.process:
            return
        proc = self.process
        self._append_log(f'[guardian] stopping pid={proc.pid}')
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            code = proc.poll()
            self.last_exit = {'time': time.time(), 'code': code if code is not None else -1}
            self.process = None

    def start(self):
        with self.lock:
            self._sync_source_policy_locked()
            self.desired = True
            if not self.process or self.process.poll() is not None:
                self._launch_locked()
        return self.status()

    def stop(self):
        with self.lock:
            self.desired = False
            self._terminate_locked()
        return self.status()

    def restart(self):
        with self.lock:
            self._sync_source_policy_locked()
            self.desired = True
            self._terminate_locked()
            self._launch_locked()
        return self.status()

    def set_auto_restart(self, enabled: bool):
        with self.lock:
            self.auto_restart = bool(enabled)
            self.auto_restart_user_set = True
            self.single_shot = bool(self.file_source and not self.auto_restart)
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
        self._append_log(f'[guardian] terminating stale inference pid={pid} reason={reason}')
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            self._append_log(f'[guardian] failed to terminate stale pid={pid}: {exc}')
            return False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._pid_exists(pid):
                self._append_log(f'[guardian] stale inference exited pid={pid}')
                return True
            time.sleep(0.1)
        force_signal = getattr(signal, 'SIGKILL', signal.SIGTERM)
        try:
            os.kill(pid, force_signal)
        except OSError as exc:
            self._append_log(f'[guardian] failed to kill stale pid={pid}: {exc}')
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
        except ConfigError as exc:
            self._append_log(f'[guardian] config error: {exc}')
            runtime = resolve_runtime_settings({}, self.config_path.parent, namespace_hint=self.config_path.stem)
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
                info['healthy'] = False
                info['reason'] = 'progress_stale'
                return info
        elif not startup_in_grace:
            info['healthy'] = False
            info['reason'] = 'progress_missing'
            return info

        info['reason'] = 'ok'
        return info

    def _monitor_loop(self):
        while not self.shutdown.is_set():
            with self.lock:
                self._sync_source_policy_locked()
                desired = self.desired
                auto_restart = self.auto_restart
                file_source = self.file_source
            if desired:
                should_launch = False
                with self.lock:
                    if not self.process:
                        should_launch = True
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
                            else:
                                should_launch = auto_restart
                        else:
                            heartbeat = self._check_heartbeat_locked()
                            self.last_heartbeat_status = heartbeat
                            if not heartbeat.get('healthy', True):
                                reason = heartbeat.get('reason', 'unknown')
                                self._append_log(f'[guardian] unhealthy heartbeat detected: {reason}')
                                self._terminate_locked()
                                if file_source and not auto_restart:
                                    self.desired = False
                                    self.single_shot = False
                                else:
                                    should_launch = auto_restart
                if should_launch:
                    try:
                        with self.lock:
                            self._launch_locked()
                    except Exception as exc:
                        self._append_log(f'[guardian] failed to start inference: {exc}')
                        time.sleep(5)
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
