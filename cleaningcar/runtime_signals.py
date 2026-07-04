from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import cv2

from .video_io import _resolve_runtime_path

DEFAULT_COMMAND_DIR = "/dev/shm/cleaningcar_cmd"
DEFAULT_HEARTBEAT_PATH = "/dev/shm/cleaningcar_heartbeat.json"
DEFAULT_STARTUP_FLAG_PATH = "/dev/shm/cleaningcar_started.flag"
DEFAULT_DEBUG_FRAME_PATH = "/dev/shm/cleaningcar_debug.jpg"
DEFAULT_RUNTIME_ROOT = "/dev/shm/cleaningcar_runtime"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_namespace_key(value: Any, default: str = "default") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._-")
    return text or default


def _resolve_namespaced_runtime_path(
    configured_value: Any,
    legacy_default: str,
    namespaced_default: Path,
    base_dir: Optional[Path],
) -> Path:
    configured_text = str(configured_value or "").strip()
    if not configured_text:
        return namespaced_default
    resolved_configured = _resolve_runtime_path(configured_text, base_dir)
    resolved_legacy = _resolve_runtime_path(legacy_default, base_dir)
    if resolved_configured == resolved_legacy:
        return namespaced_default
    return resolved_configured


def _resolve_optional_debug_frame_path(
    configured_value: Any,
    namespaced_default: Path,
    base_dir: Optional[Path],
) -> Optional[Path]:
    if isinstance(configured_value, bool):
        if not configured_value:
            return None
        configured_text = DEFAULT_DEBUG_FRAME_PATH
    else:
        configured_text = str(configured_value or "").strip()
    # Empty means "use the namespaced default debug frame path".
    if not configured_text:
        return namespaced_default
    if configured_text.lower() in ("0", "false", "none", "null", "off", "disable", "disabled"):
        return None
    return _resolve_namespaced_runtime_path(
        configured_text,
        DEFAULT_DEBUG_FRAME_PATH,
        namespaced_default,
        base_dir,
    )


def resolve_runtime_settings(
    config: Optional[Dict[str, Any]],
    base_dir: Optional[Path],
    namespace_hint: Optional[str] = None,
) -> Dict[str, Any]:
    system_cfg = (config or {}).get("system", {}) or {}
    device_id = str(system_cfg.get("device_id", "") or "").strip()
    namespace_source = device_id or namespace_hint or (config or {}).get("config_name") or "default"
    runtime_namespace_key = _safe_namespace_key(namespace_source)
    runtime_root = _resolve_runtime_path(DEFAULT_RUNTIME_ROOT, base_dir) / runtime_namespace_key
    heartbeat_interval = max(0.5, _safe_float(system_cfg.get("heartbeat_interval_seconds"), 1.0))
    heartbeat_timeout = max(5.0, _safe_float(system_cfg.get("heartbeat_timeout_seconds"), 30.0))
    progress_timeout = max(heartbeat_timeout, _safe_float(system_cfg.get("progress_timeout_seconds"), 90.0))
    startup_grace = max(progress_timeout, _safe_float(system_cfg.get("heartbeat_startup_grace_seconds"), 90.0))
    return {
        "device_id": device_id,
        "runtime_namespace_key": runtime_namespace_key,
        "runtime_root": runtime_root,
        "command_dir": _resolve_namespaced_runtime_path(
            system_cfg.get("command_dir", DEFAULT_COMMAND_DIR),
            DEFAULT_COMMAND_DIR,
            runtime_root / "cmd",
            base_dir,
        ),
        "heartbeat_path": _resolve_namespaced_runtime_path(
            system_cfg.get("heartbeat_path", DEFAULT_HEARTBEAT_PATH),
            DEFAULT_HEARTBEAT_PATH,
            runtime_root / "heartbeat.json",
            base_dir,
        ),
        "startup_flag_path": _resolve_namespaced_runtime_path(
            system_cfg.get("startup_flag_path", DEFAULT_STARTUP_FLAG_PATH),
            DEFAULT_STARTUP_FLAG_PATH,
            runtime_root / "started.flag",
            base_dir,
        ),
        "startup_capture_dir": _resolve_runtime_path(
            system_cfg.get("startup_capture_dir", "captures/startup"),
            base_dir,
        ),
        "manual_capture_dir": _resolve_runtime_path(
            system_cfg.get("manual_capture_dir", "captures/manual"),
            base_dir,
        ),
        "debug_frame_path": _resolve_optional_debug_frame_path(
            ((config or {}).get("video", {}) or {}).get("debug_frame_path", DEFAULT_DEBUG_FRAME_PATH),
            runtime_root / "debug.jpg",
            base_dir,
        ),
        "heartbeat_interval_seconds": heartbeat_interval,
        "heartbeat_timeout_seconds": heartbeat_timeout,
        "progress_timeout_seconds": progress_timeout,
        "heartbeat_startup_grace_seconds": startup_grace,
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_snapshot_command(command_dir: Path, payload: Dict[str, Any], prefix: str = "snapshot") -> Path:
    command_dir = Path(command_dir)
    command_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"{prefix}_{ts}_{uuid4().hex[:8]}.cmd.json"
    command_path = command_dir / filename
    write_json_atomic(command_path, payload)
    return command_path


def _safe_tag(tag: Any, default: str = "manual") -> str:
    text = str(tag or "").strip()
    if not text:
        return default
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", text)
    text = text.strip("_")
    return text or default


def _overlay_signal_text(frame, text: str):
    if frame is None:
        return None
    output = frame.copy()
    h, w = output.shape[:2]
    scale = max(0.8, min(w, h) / 900.0)
    thickness_bg = max(3, int(scale * 5))
    thickness_fg = max(1, int(scale * 2))
    org = (20, max(40, int(50 * scale)))
    cv2.putText(
        output,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness_bg,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness_fg,
        cv2.LINE_AA,
    )
    return output


def save_snapshot_images(
    root_dir: Path,
    frame_idx: int,
    raw_frame,
    annotated_frame,
    capture_ts: Optional[float] = None,
    tag: str = "manual",
    save_raw: bool = True,
    save_annotated: bool = True,
    signal_text: str = "",
) -> Dict[str, str]:
    now_ts = float(capture_ts) if capture_ts is not None else time.time()
    dt = time.localtime(now_ts)
    day_dir = time.strftime("%Y%m%d", dt)
    hour_dir = time.strftime("%H", dt)
    target_dir = Path(root_dir) / day_dir / hour_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", dt)
    safe_tag = _safe_tag(tag)
    base_name = f"{stamp}_f{int(frame_idx):06d}_{safe_tag}"
    saved: Dict[str, str] = {}

    if save_raw and raw_frame is not None:
        raw_path = target_dir / f"{base_name}_raw.jpg"
        if cv2.imwrite(str(raw_path), raw_frame):
            saved["raw"] = str(raw_path)

    if save_annotated:
        render_frame = annotated_frame if annotated_frame is not None else raw_frame
        if render_frame is not None:
            if signal_text:
                render_frame = _overlay_signal_text(render_frame, signal_text)
            ann_path = target_dir / f"{base_name}_annotated.jpg"
            if cv2.imwrite(str(ann_path), render_frame):
                saved["annotated"] = str(ann_path)

    return saved
