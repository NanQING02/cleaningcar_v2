from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from . import state
from .helpers import _load_config, _resolve_config_path
from .inference import _get_inference_manager_for_key


router = APIRouter(prefix="/workbench", tags=["workbench"])


def _template(name: str) -> str:
    path = state.ROOT / "web" / "templates" / name
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"工作台模板缺失: {name}")
    return path.read_text(encoding="utf-8")


def _resolve_path(value: Any, base_dir: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _event_dir(key: Optional[str]) -> Path:
    cfg = _load_config(key)
    configured = cfg.data.get("event_output_dir") or cfg.data.get("system", {}).get("event_output_dir")
    return _resolve_path(configured, state.ROOT) or (state.ROOT / "events")


def _event_dirs(key: Optional[str]) -> list[Path]:
    if key != "__all__":
        return [_event_dir(key)]
    dirs = []
    for config_path in state.CONFIG_PATH.parent.glob("*.json"):
        try:
            cfg = _load_config(config_path.name)
            directory = _resolve_path(
                cfg.data.get("event_output_dir") or cfg.data.get("system", {}).get("event_output_dir"),
                state.ROOT,
            ) or (state.ROOT / "events")
            if directory not in dirs:
                dirs.append(directory)
        except Exception:
            continue
    return dirs or [state.ROOT / "events"]


def _capture_dirs(key: Optional[str]) -> list[Path]:
    if key == "__all__":
        dirs = []
        for config_path in state.CONFIG_PATH.parent.glob("*.json"):
            for path in _capture_dirs(config_path.name):
                if path not in dirs:
                    dirs.append(path)
        return dirs
    cfg = _load_config(key)
    dirs = [_event_dir(key)]
    for value in (
        cfg.data.get("event_capture_dir"),
        cfg.data.get("logic", {}).get("per_id_video_dir"),
        cfg.data.get("system", {}).get("manual_capture_dir"),
        cfg.data.get("system", {}).get("startup_capture_dir"),
    ):
        path = _resolve_path(value, state.ROOT)
        if path:
            dirs.append(path)
    return dirs


def _allowed_file(path: Path, key: Optional[str]) -> bool:
    try:
        candidate = path.resolve()
    except OSError:
        return False
    for base in _capture_dirs(key):
        try:
            candidate.relative_to(base.resolve())
            return True
        except ValueError:
            continue
    return False


def _read_events(key: Optional[str]) -> list[dict]:
    grouped: Dict[str, dict] = {}
    config_name = _load_config(None if key == "__all__" else key).path.name
    device_to_config = {}
    for config_path in state.CONFIG_PATH.parent.glob("*.json"):
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            device_id = str(config_data.get("system", {}).get("device_id", "") or "").strip()
            if device_id:
                device_to_config[device_id] = config_path.name
        except (OSError, ValueError):
            continue
    paths = []
    for directory in _event_dirs(key):
        if directory.exists():
            paths.extend(directory.glob("*.json"))
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "type" not in data or "id" not in data:
            continue
        event_id = str(data.get("id") or path.stem)
        event_config_name = config_name
        for known_device_id, known_config_name in sorted(device_to_config.items(), key=lambda item: len(item[0]), reverse=True):
            if event_id == known_device_id or event_id.startswith(known_device_id + "-"):
                event_config_name = known_config_name
                break
        item = grouped.setdefault(
            event_id,
            {
                "id": event_id,
                "plateNumber": "",
                "vehicleType": "",
                "plateColor": "",
                "lane": "",
                "captureTime": "",
                "captureImage": "",
                "stages": [],
                "abnormal": False,
                "abnormalReason": "",
                "sourceFiles": [],
                "stageAbnormalReasons": [],
                "configName": event_config_name,
            },
        )
        event_type = int(data.get("type") or 0)
        item["stages"].append(
            {
                "type": event_type,
                "captureTime": data.get("captureTime", ""),
                "captureImage": data.get("captureImage", ""),
                "washDuration": data.get("washDuration", 0.0),
                "sourceFile": str(path),
                "isAbnormal": bool(data.get("isAbnormal") or data.get("plateRecognitionAbnormal")),
                "abnormalReason": data.get("abnormalReason", ""),
            }
        )
        item["sourceFiles"].append(str(path))
        if event_type >= int(item.get("latestType", 0)):
            item["latestType"] = event_type
            for field in ("plateNumber", "vehicleType", "plateColor", "lane", "captureTime", "captureImage"):
                if data.get(field) not in (None, ""):
                    item[field] = data.get(field)
            item["washDuration"] = data.get("totalWashDuration", data.get("washDuration", 0.0))
            item["directionLabel"] = data.get("directionLabel", "")
            item["wheelResults"] = data.get("wheelResults", [])
        reason = str(data.get("abnormalReason") or "").strip()
        if reason:
            item["stageAbnormalReasons"].append({"type": event_type, "reason": reason})
        if event_type >= int(item.get("latestType", 0)):
            item["abnormal"] = bool(data.get("isAbnormal") or data.get("plateRecognitionAbnormal"))
            item["abnormalReason"] = reason
    result = []
    for item in grouped.values():
        item["stages"].sort(key=lambda stage: (stage.get("captureTime", ""), stage.get("type", 0)))
        item["status"] = "已结束" if item.get("latestType", 0) >= 5 else "进行中"
        item["reviewStatus"] = "待复核" if item["abnormal"] else "未复核"
        item["uploadStatus"] = "待确认"
        item["configName"] = item.get("configName") or config_name
        result.append(item)
    result.sort(key=lambda item: item.get("captureTime", ""), reverse=True)
    return result


@router.get("", response_class=HTMLResponse)
def workbench_page():
    return _template("workbench.html")


@router.get("/settings", response_class=HTMLResponse)
def workbench_settings_page():
    return _template("workbench_settings.html")


@router.get("/events")
def workbench_events(
    key: Optional[str] = None,
    query: str = "",
    status: str = "",
    abnormal: str = "",
    config_name: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    rows = _read_events(key)
    if config_name:
        rows = [row for row in rows if row.get("configName") == config_name]
    query = query.strip().lower()
    if query:
        rows = [
            row
            for row in rows
            if query in str(row.get("plateNumber", "")).lower()
            or query in str(row.get("vehicleType", "")).lower()
            or query in str(row.get("id", "")).lower()
        ]
    if status:
        rows = [row for row in rows if row.get("status") == status]
    if abnormal == "yes":
        rows = [row for row in rows if row.get("abnormal")]
    elif abnormal == "no":
        rows = [row for row in rows if not row.get("abnormal")]
    total = len(rows)
    start = (page - 1) * page_size
    return {"items": rows[start : start + page_size], "page": page, "page_size": page_size, "total": total}


@router.get("/events/{event_id}")
def workbench_event_detail(event_id: str, key: Optional[str] = None):
    for row in _read_events(key):
        if str(row.get("id")) == str(event_id):
            return row
    raise HTTPException(status_code=404, detail="事件不存在")


@router.get("/media")
def workbench_media(path: str, key: Optional[str] = None):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (state.ROOT / candidate).resolve()
    if not _allowed_file(candidate, key) or not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="媒体文件不存在或不允许访问")
    return FileResponse(candidate)


@router.get("/runtime")
def workbench_runtime(key: Optional[str] = None):
    manager = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    cfg = _load_config(key)
    return {
        "config": cfg.path.name,
        "device_id": cfg.data.get("system", {}).get("device_id", ""),
        "lane": cfg.data.get("logic", {}).get("lane_name", ""),
        "inference": manager.status(),
        "events": {"count": len(_read_events(key))},
    }
