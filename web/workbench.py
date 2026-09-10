from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from cleaningcar.report_agent import (
    ReportAgentError,
    ReportAgentSettings,
    build_compact_event,
    stream_report,
)

from . import state
from .helpers import _load_config, _resolve_config_path
from .inference import _get_inference_manager_for_key


router = APIRouter(prefix="/workbench", tags=["workbench"])

EVENT_STAGE_META = {
    1: ("进入检测区 A", "车辆进入本次过车生命周期"),
    2: ("进入冲洗区 B", "冲洗过程窗口开始"),
    3: ("检测到水流", "B 区内出现冲水证据"),
    4: ("退出冲洗区 B", "冲洗过程窗口结束"),
    5: ("退出检测区 A", "最终业务记录形成"),
    6: ("单车录像结束", "录像生命周期结束通知，不属于冲洗过程"),
}


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


def _per_id_video_dirs(key: Optional[str]) -> list[Path]:
    if key == "__all__":
        dirs = []
        for config_path in state.CONFIG_PATH.parent.glob("*.json"):
            for path in _per_id_video_dirs(config_path.name):
                if path not in dirs:
                    dirs.append(path)
        return dirs
    cfg = _load_config(key)
    configured = _resolve_path(cfg.data.get("logic", {}).get("per_id_video_dir"), state.ROOT)
    dirs = [path for path in (configured, state.ROOT / "video_result" / "per_id") if path]
    return list(dict.fromkeys(dirs))


def _find_per_id_video(event_id: str, key: Optional[str]) -> str:
    filename = f"{event_id}.mp4"
    for directory in _per_id_video_dirs(key):
        if not directory.exists():
            continue
        direct = directory / filename
        if direct.exists() and direct.is_file():
            return str(direct)
        try:
            match = next(directory.rglob(filename), None)
        except OSError:
            match = None
        if match and match.is_file():
            return str(match)
    return ""


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
                "latestType": 0,
                "latestBusinessType": 0,
            },
        )
        event_type = int(data.get("type") or 0)
        stage_label, stage_description = EVENT_STAGE_META.get(event_type, (f"Type {event_type}", "未知事件阶段"))
        item["stages"].append(
            {
                "type": event_type,
                "label": stage_label,
                "description": stage_description,
                "captureTime": data.get("captureTime", ""),
                "captureImage": data.get("captureImage", ""),
                "washDuration": data.get("washDuration", 0.0),
                "washStartTime": data.get("washStartTime", ""),
                "totalWashDuration": data.get("totalWashDuration"),
                "direction": data.get("direction"),
                "directionLabel": data.get("directionLabel", ""),
                "directionSource": data.get("directionSource", ""),
                "videoEndTime": data.get("videoEndTime", ""),
                "videoDuration": data.get("videoDuration"),
                "perIdVideoEnabled": data.get("perIdVideoEnabled"),
                "sourceFile": str(path),
                "isAbnormal": bool(data.get("isAbnormal") or data.get("plateRecognitionAbnormal")),
                "abnormalReason": data.get("abnormalReason", ""),
            }
        )
        item["sourceFiles"].append(str(path))
        item["latestType"] = max(event_type, int(item.get("latestType", 0)))
        if 1 <= event_type <= 5 and event_type >= int(item.get("latestBusinessType", 0)):
            item["latestBusinessType"] = event_type
            for field in ("plateNumber", "vehicleType", "plateColor", "lane", "captureTime", "captureImage"):
                if data.get(field) not in (None, ""):
                    item[field] = data.get(field)
            item["washDuration"] = data.get("totalWashDuration", data.get("washDuration", 0.0))
            item["directionLabel"] = data.get("directionLabel", "")
            item["wheelResults"] = data.get("wheelResults", [])
        reason = str(data.get("abnormalReason") or "").strip()
        if reason:
            item["stageAbnormalReasons"].append({"type": event_type, "reason": reason})
        if 1 <= event_type <= 5 and event_type >= int(item.get("latestBusinessType", 0)):
            item["abnormal"] = bool(data.get("isAbnormal") or data.get("plateRecognitionAbnormal"))
            item["abnormalReason"] = reason
    result = []
    for item in grouped.values():
        item["stages"].sort(key=lambda stage: (stage.get("captureTime", ""), stage.get("type", 0)))
        stage_by_type = {stage.get("type"): stage for stage in item["stages"]}
        item["process"] = {
            "zoneAEnterTime": (stage_by_type.get(1) or {}).get("captureTime", ""),
            "zoneBEnterTime": (stage_by_type.get(2) or {}).get("captureTime", ""),
            "waterDetectTime": (stage_by_type.get(3) or {}).get("captureTime", ""),
            "waterDetected": 3 in stage_by_type,
            "zoneBExitTime": (stage_by_type.get(4) or {}).get("captureTime", ""),
            "zoneAExitTime": (stage_by_type.get(5) or {}).get("captureTime", ""),
            "videoCompleteTime": (stage_by_type.get(6) or {}).get("captureTime", ""),
        }
        item["status"] = "已结束" if any(stage.get("type") == 5 for stage in item["stages"]) else "进行中"
        item["reviewStatus"] = "待复核" if item["abnormal"] else "未复核"
        item["uploadStatus"] = "待确认"
        item["configName"] = item.get("configName") or config_name
        result.append(item)
    result.sort(key=lambda item: item.get("captureTime", ""), reverse=True)
    return result


def _find_event(event_id: str, key: Optional[str]) -> dict:
    for row in _read_events(key):
        if str(row.get("id")) == str(event_id):
            return row
    raise HTTPException(status_code=404, detail="事件不存在")


def _event_config(row: dict, key: Optional[str]):
    if key == "__all__":
        return _load_config(str(row.get("configName") or ""))
    return _load_config(key)


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _report_cache_path(cfg, event_id: str) -> Path:
    event_dir = cfg.data.get("event_output_dir") or cfg.data.get("system", {}).get("event_output_dir")
    directory = (_resolve_path(event_dir, state.ROOT) or (state.ROOT / "events")) / "agent_reports"
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(event_id))[:80]
    digest = hashlib.sha256(str(event_id).encode("utf-8", errors="replace")).hexdigest()[:12]
    return directory / f"{safe_prefix or 'record'}_{digest}.json"


def _read_cached_report(cfg, event_id: str) -> Optional[dict]:
    path = _report_cache_path(cfg, event_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    report = str(data.get("report") or "") if isinstance(data, dict) else ""
    if not report:
        return None
    return {
        "available": True,
        "report": report,
        "generatedAt": str(data.get("generatedAt") or ""),
        "model": str(data.get("model") or ""),
    }


def _write_cached_report(cfg, event_id: str, report: str, model: str) -> None:
    path = _report_cache_path(cfg, event_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recordId": str(event_id),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model),
        "report": str(report),
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


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
    row = _find_event(event_id, key)
    row["videoPath"] = _find_per_id_video(event_id, key)
    return row


@router.get("/events/{event_id}/agent/status")
def workbench_agent_status(event_id: str, key: Optional[str] = None):
    row = _find_event(event_id, key)
    settings = ReportAgentSettings.from_config(_event_config(row, key).data)
    return {**settings.public_status(), "record_complete": row.get("status") == "已结束"}


@router.get("/events/{event_id}/agent/report")
def workbench_cached_agent_report(event_id: str, key: Optional[str] = None):
    row = _find_event(event_id, key)
    cached = _read_cached_report(_event_config(row, key), event_id)
    return cached or {"available": False}


@router.post("/events/{event_id}/agent/report")
def workbench_agent_report(event_id: str, key: Optional[str] = None):
    row = _find_event(event_id, key)
    if row.get("status") != "已结束":
        raise HTTPException(status_code=409, detail="过车记录尚未结束，暂不能生成完整报告。")
    cfg = _event_config(row, key)
    settings = ReportAgentSettings.from_config(cfg.data)
    try:
        settings.validate()
    except ReportAgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    compact = build_compact_event(row)

    def generate():
        yield _sse("meta", {"record": compact, "model": settings.model})
        produced = False
        report_parts = []
        try:
            for text in stream_report(compact, settings):
                produced = True
                report_parts.append(text)
                yield _sse("token", {"text": text})
            if not produced:
                yield _sse("error", {"message": "大模型未返回报告内容。"})
                return
            _write_cached_report(cfg, event_id, "".join(report_parts), settings.model)
            yield _sse("done", {"status": "ok"})
        except ReportAgentError as exc:
            yield _sse("error", {"message": str(exc)})
        except Exception:
            yield _sse("error", {"message": "报告生成失败，请查看 Web 服务日志。"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
