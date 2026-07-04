import csv
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import HTTPException

from cleaningcar.runtime_signals import resolve_runtime_settings
from config_manager import ConfigError, ConfigManager

from . import state


def _resolve_path(path_str: Optional[str], default: Optional[Path] = None) -> Path:
    if path_str:
        candidate = Path(path_str)
    elif default is not None:
        candidate = Path(default)
    else:
        candidate = state.ROOT
    if not candidate.is_absolute():
        candidate = (state.ROOT / candidate).resolve()
    return candidate


def _resolve_config_path(key: Optional[str] = None) -> Path:
    key = str(key or "").strip()
    if not key:
        return state.CONFIG_PATH
    candidate = (state.CONFIG_PATH.parent / key).resolve()
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")
    if candidate.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="仅支持 JSON 配置")
    return candidate


def _event_log_path(cfg: ConfigManager) -> Path:
    base_dir = cfg.data.get("event_output_dir")
    if not base_dir:
        base_dir = cfg.data.get("system", {}).get("event_output_dir")
    base = _resolve_path(base_dir or (state.ROOT / "events"))
    return base / "event_log.csv"


def _detection_csv_path(cfg: ConfigManager) -> Optional[Path]:
    csv_path = cfg.video.get("csv")
    if not csv_path:
        return None
    return _resolve_path(csv_path)


def _debug_frame_path(cfg: ConfigManager) -> Optional[Path]:
    runtime = resolve_runtime_settings(cfg.data, cfg.path.parent, namespace_hint=cfg.path.stem)
    dbg_path = runtime.get("debug_frame_path")
    if not dbg_path:
        return None
    return Path(dbg_path)


def _read_csv_tail(path: Path, limit: int):
    if not path.exists():
        return []
    rows = deque(maxlen=max(1, limit))
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return list(rows)


def _load_config(key: Optional[str] = None) -> ConfigManager:
    cfg_path = _resolve_config_path(key)
    try:
        return ConfigManager(cfg_path)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _inference_log_dir() -> Path:
    return state.ROOT / "logs" / "inference"


def _capture_frame(force: bool = False, key: Optional[str] = None):
    cache_key = str(_resolve_config_path(key))
    if state.FRAME_CACHE.data is not None and state.FRAME_CACHE.key == cache_key and not force:
        return
    cfg = _load_config(key)
    src = cfg.video.get("source")
    if not src:
        raise HTTPException(status_code=400, detail="video.source 未配置")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"无法打开视频源: {src}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="视频源无法读取帧")
    success, buf = cv2.imencode(".jpg", frame)
    if not success:
        raise HTTPException(status_code=500, detail="帧编码失败")
    state.FRAME_CACHE.data = buf.tobytes()
    state.FRAME_CACHE.size = (frame.shape[1], frame.shape[0])
    state.FRAME_CACHE.key = cache_key


def _available_config_files() -> List[Path]:
    base = state.CONFIG_PATH.parent
    files = [path for path in base.glob("*.json") if path.is_file()]
    return sorted(files)
