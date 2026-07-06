from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT_PATH = PROJECT_ROOT / "tools" / "rk3588_fixed_freq.sh"


def _parse_bool_like(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _logger_emit(logger: Optional[Callable[[str], None]], message: str) -> None:
    if callable(logger):
        logger(message)
    else:
        print(message)


def _system_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    system = config.get("system")
    return system if isinstance(system, dict) else {}


def perf_lock_enabled(config: Optional[Dict[str, Any]]) -> bool:
    env_value = _parse_bool_like(os.environ.get("CLEANINGCAR_PERF_LOCK"))
    if env_value is not None:
        return bool(env_value)
    cfg_value = _parse_bool_like(_system_config(config).get("performance_lock_enabled"))
    if cfg_value is not None:
        return bool(cfg_value)
    return True


def perf_lock_restore_enabled(config: Optional[Dict[str, Any]]) -> bool:
    env_value = _parse_bool_like(os.environ.get("CLEANINGCAR_PERF_LOCK_RESTORE_ON_STOP"))
    if env_value is not None:
        return bool(env_value)
    cfg_value = _parse_bool_like(_system_config(config).get("performance_lock_restore_on_stop"))
    return bool(cfg_value)


def perf_lock_script_path(config: Optional[Dict[str, Any]]) -> Path:
    env_path = str(os.environ.get("CLEANINGCAR_PERF_LOCK_SCRIPT", "") or "").strip()
    if env_path:
        return Path(env_path).expanduser()
    cfg_path = str(_system_config(config).get("performance_lock_script", "") or "").strip()
    if cfg_path:
        return Path(cfg_path).expanduser()
    return DEFAULT_SCRIPT_PATH


def _use_sudo() -> bool:
    env_value = _parse_bool_like(os.environ.get("CLEANINGCAR_PERF_LOCK_USE_SUDO"))
    if env_value is not None:
        return bool(env_value)
    return True


def build_perf_lock_command(script_path: Path, action: str) -> list[str]:
    script = Path(script_path).expanduser().resolve()
    if os.geteuid() == 0:
        return ["bash", str(script), action]
    if _use_sudo() and shutil.which("sudo"):
        return ["sudo", "-n", "bash", str(script), action]
    return ["bash", str(script), action]


def run_perf_lock_action(
    action: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    path = perf_lock_script_path(config)
    if not path.exists():
        message = f"[perf-lock] script not found: {path}"
        _logger_emit(logger, message)
        return {
            "attempted": False,
            "ok": False,
            "path": str(path),
            "action": action,
            "reason": "missing_script",
            "message": message,
        }

    cmd = build_perf_lock_command(path, action)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except Exception as exc:
        message = f"[perf-lock] action={action} failed to start: {exc}"
        _logger_emit(logger, message)
        return {
            "attempted": True,
            "ok": False,
            "path": str(path),
            "action": action,
            "reason": "exec_failed",
            "message": message,
        }

    stdout = str(result.stdout or "").strip()
    stderr = str(result.stderr or "").strip()
    ok = result.returncode == 0
    summary = (
        f"[perf-lock] action={action} rc={result.returncode} path={path}"
        + (f" stdout={stdout}" if stdout else "")
        + (f" stderr={stderr}" if stderr else "")
    )
    _logger_emit(logger, summary)
    return {
        "attempted": True,
        "ok": ok,
        "path": str(path),
        "action": action,
        "returncode": int(result.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "message": summary,
    }


def maybe_apply_perf_lock(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not perf_lock_enabled(config):
        return {
            "attempted": False,
            "ok": False,
            "action": "apply",
            "reason": "disabled",
        }
    return run_perf_lock_action("apply", config=config, logger=logger)


def maybe_restore_perf_lock(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not perf_lock_restore_enabled(config):
        return {
            "attempted": False,
            "ok": False,
            "action": "restore",
            "reason": "disabled",
        }
    return run_perf_lock_action("restore", config=config, logger=logger)
