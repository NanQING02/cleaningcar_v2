import argparse
import json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from starlette.responses import JSONResponse as StarletteJSONResponse


def _sanitize_surrogates(obj: Any) -> Any:
    """Replace lone surrogates (U+D800–U+DFFF) with U+FFFD so JSON encoding succeeds.

    Lone surrogates can leak into Python strings via PEP 383 surrogateescape
    when filesystem paths contain non-UTF-8 bytes — the classic failure is
    ``json.dumps(..., ensure_ascii=False).encode('utf-8')`` raising
    ``UnicodeEncodeError: surrogates not allowed``.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {_sanitize_surrogates(k): _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_surrogates(item) for item in obj]
    return obj


class SafeJSONResponse(StarletteJSONResponse):
    """JSONResponse that sanitises lone-surrogate strings before rendering."""

    def render(self, content: Any) -> bytes:
        return super().render(_sanitize_surrogates(content))

from cleaningcar.runtime_signals import resolve_runtime_settings, write_snapshot_command

from . import state
from .config_tiers import (
    ConfigTierError,
    TIER_DEVELOPER,
    TIER_USER,
    extract_tier_subset,
    merge_tier_payload,
    schema_payload,
)
from .helpers import (
    _available_config_files,
    _capture_frame,
    _debug_frame_path,
    _detection_csv_path,
    _event_log_path,
    _event_output_dir,
    _inference_log_dir,
    _load_config,
    _resolve_config_path,
    _read_csv_tail,
)
from utils.upload_queue import SQLiteUploadQueue
from .inference import (
    _get_default_inference_manager,
    _get_inference_manager_for_key,
    startup_default_manager,
    shutdown_all_inference_managers,
)
from .models import (
    ConfigPayload,
    ConfigSaveAsPayload,
    ConfigSelectPayload,
    SnapshotKeepPayload,
    ZonePayload,
)


def _payload_dict(payload: ConfigPayload) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=True)
    return payload.dict(exclude_none=True)


def _extract_device_id(config_data: dict) -> str:
    system_cfg = (config_data or {}).get("system", {}) or {}
    return str(system_cfg.get("device_id", "") or "").strip()


def _validate_device_id_unique(config_path: Path, config_data: dict) -> None:
    device_id = _extract_device_id(config_data)
    if not device_id:
        return
    duplicates = []
    target_path = Path(config_path).resolve()
    for candidate in sorted(target_path.parent.glob("*.json")):
        try:
            candidate_path = candidate.resolve()
        except OSError:
            continue
        if candidate_path == target_path:
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                other_data = json.load(f)
        except Exception:
            continue
        if _extract_device_id(other_data) == device_id:
            duplicates.append(candidate.name)
    if duplicates:
        joined = "、".join(duplicates)
        raise HTTPException(
            status_code=400,
            detail=f"device_id '{device_id}' 与其他配置重复：{joined}。多路运行时每份配置的 device_id 必须唯一。",
        )


def _apply_config_update(payload: dict, tier: Optional[str] = None, key: Optional[str] = None):
    cfg = _load_config(key)
    try:
        merge_tier_payload(cfg.data, payload, tier)
        _validate_device_id_unique(cfg.path, cfg.data)
        cfg.save()
    except ConfigTierError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存失败: {exc}") from exc
    if any(item in payload for item in ("video", "zones")):
        state.FRAME_CACHE.clear()
    return {"status": "ok", "tier": tier or "all", "config": cfg.path.name}

@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    try:
        startup_default_manager()
    except Exception as exc:
        print(f'[server] startup: failed to init default inference manager: {exc}')
    try:
        yield
    finally:
        shutdown_all_inference_managers()


app = FastAPI(title='CleaningCar Zone Editor', lifespan=_app_lifespan, default_response_class=SafeJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/", response_class=HTMLResponse)
def index():
    if not state.TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="Template missing.")
    return state.TEMPLATE_PATH.read_text(encoding="utf-8")


@app.get("/static/vue.global.prod.js")
def vue_bundle():
    path = state.ROOT / "web" / "static" / "vue.global.prod.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Vue bundle missing.")
    return FileResponse(path)


@app.get("/static/{name}")
def static_file(name: str):
    path = state.ROOT / "web" / "static" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Static file missing.")
    return FileResponse(path)


@app.get("/zones")
def read_zones(key: Optional[str] = None):
    cfg = _load_config(key)
    return cfg.zones


@app.post("/zones")
def update_zones(payload: ZonePayload, key: Optional[str] = None):
    cfg = _load_config(key)
    reference_edge = payload.direction_reference_edge
    if isinstance(reference_edge, str) and reference_edge.strip().lower() == "auto":
        reference_edge = "auto"
    else:
        try:
            reference_edge = int(reference_edge)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="方向参考边必须为 auto 或有效边序号") from exc
        if not 0 <= reference_edge < len(payload.zone_a_detection):
            raise HTTPException(status_code=400, detail="方向参考边超出 Zone A 边数")
    cfg.data.setdefault("zones", {})
    cfg.data["zones"]["zone_a_detection"] = payload.zone_a_detection
    cfg.data["zones"]["zone_b_wash"] = payload.zone_b_wash
    cfg.data["zones"]["flow_vector"] = {
        "start": payload.flow_vector.start,
        "end": payload.flow_vector.end,
    }
    cfg.data["zones"]["direction_reference_edge"] = reference_edge
    cfg.save()
    return {"status": "ok"}


@app.get("/frame_meta")
def frame_meta(key: Optional[str] = None):
    if state.FRAME_CACHE.data is None or state.FRAME_CACHE.key != str(_resolve_config_path(key)):
        try:
            _capture_frame(key=key)
        except HTTPException:
            return {"available": False}
    w, h = state.FRAME_CACHE.size
    return {"available": True, "width": w, "height": h}


@app.get("/frame")
def get_frame(key: Optional[str] = None):
    if state.FRAME_CACHE.data is None or state.FRAME_CACHE.key != str(_resolve_config_path(key)):
        _capture_frame(key=key)
    return Response(content=state.FRAME_CACHE.data, media_type="image/jpeg")


@app.post("/frame/reload")
def reload_frame(key: Optional[str] = None):
    _capture_frame(force=True, key=key)
    return {"status": "ok"}


@app.get("/debug_frame_meta")
def debug_frame_meta(key: Optional[str] = None):
    cfg = _load_config(key)
    path = _debug_frame_path(cfg)
    if not path:
        return {"available": False, "enabled": False, "path": ""}
    if not path.exists():
        return {"available": False, "enabled": True, "path": str(path)}
    try:
        stat = path.stat()
        return {"available": True, "enabled": True, "path": str(path), "updated": stat.st_mtime}
    except Exception:
        return {"available": True, "enabled": True, "path": str(path), "updated": 0}


@app.get("/debug_frame")
def debug_frame(key: Optional[str] = None):
    cfg = _load_config(key)
    path = _debug_frame_path(cfg)
    if not path:
        raise HTTPException(status_code=404, detail="调试帧已禁用，请设置 debug_frame_path。")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"调试帧尚未生成，请确认推理已运行并等待输出：{path}")
    try:
        data = path.read_bytes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法读取调试帧: {exc}") from exc
    return Response(content=data, media_type="image/jpeg")


@app.get("/config")
def read_config(key: Optional[str] = None):
    cfg = _load_config(key)
    return cfg.data


@app.get("/config/schema")
def read_config_schema():
    return schema_payload()


@app.get("/config/user")
def read_user_config(key: Optional[str] = None):
    cfg = _load_config(key)
    return extract_tier_subset(cfg.data, TIER_USER)


@app.post("/config/save_as")

def save_config_as(payload: ConfigSaveAsPayload):
    name = str(payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if any(sep in name for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail="文件名不能包含路径分隔符")
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    base = state.CONFIG_PATH.parent
    target = (base / name).resolve()
    if target.exists():
        raise HTTPException(status_code=409, detail="配置文件已存在")
    try:
        _validate_device_id_unique(target, payload.data)
        with target.open("w", encoding="utf-8") as f:
            json.dump(_sanitize_surrogates(payload.data), f, ensure_ascii=False, indent=2)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存失败: {exc}") from exc
    return {"saved_as": target.name}

@app.get("/config/files")
def list_config_files():
    files = _available_config_files()
    device_ids = {}
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            device_ids[path.name] = _extract_device_id(data)
        except Exception:
            device_ids[path.name] = ""
    return {
        "active": state.CONFIG_PATH.name,
        "files": [p.name for p in files],
        "device_ids": device_ids,
    }


@app.post("/config/select")
def select_config(payload: ConfigSelectPayload):
    base = state.CONFIG_PATH.parent
    candidate = (base / payload.name).resolve()
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")
    if candidate.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="仅支持 JSON 配置")
    state.set_config_path(candidate)
    mgr = _get_default_inference_manager()
    mgr.set_config_path(candidate)
    state.FRAME_CACHE.clear()
    return {"active": candidate.name}

@app.post("/config")
def update_config(payload: ConfigPayload, key: Optional[str] = None):
    return _apply_config_update(_payload_dict(payload), key=key)


@app.post("/config/user")
def update_user_config(payload: ConfigPayload, key: Optional[str] = None):
    return _apply_config_update(_payload_dict(payload), TIER_USER, key=key)


@app.post("/config/developer")
def update_developer_config(payload: ConfigPayload, key: Optional[str] = None):
    return _apply_config_update(_payload_dict(payload), TIER_DEVELOPER, key=key)


@app.post("/inference/start")
def start_inference(key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    return mgr.start()


@app.post("/inference/stop")
def stop_inference(key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    return mgr.stop()


@app.post("/inference/restart")
def restart_inference(key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    return mgr.restart()


@app.post("/inference/auto_restart")
def set_auto_restart(enable: bool = True, key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    mgr.set_auto_restart(enable)
    return mgr.status()


@app.get("/inference/status")
def inference_status(key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    return mgr.status()


@app.get("/inference/health")
def inference_health(key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    return mgr.status()


@app.post("/snapshot/keep")
def keep_snapshot(payload: SnapshotKeepPayload, key: Optional[str] = None):
    if not payload.raw and not payload.annotated:
        raise HTTPException(status_code=400, detail="At least one of raw/annotated must be true.")
    cfg = _load_config(key)
    runtime_settings = resolve_runtime_settings(cfg.data, cfg.path.parent, namespace_hint=cfg.path.stem)
    command_path = write_snapshot_command(
        runtime_settings['command_dir'],
        {
            'cmd': 'keep_snapshot',
            'tag': payload.tag or 'manual',
            'raw': bool(payload.raw),
            'annotated': bool(payload.annotated),
            'requested_at': datetime.now().isoformat(timespec='seconds'),
            'requested_config': cfg.path.name,
            'requested_by': 'web_api',
        },
    )
    return {
        'status': 'queued',
        'command_file': str(command_path),
    }


@app.get("/logs/inference")
def inference_logs(lines: int = 200, key: Optional[str] = None):
    mgr = _get_inference_manager_for_key(key or state.CONFIG_PATH.name)
    return {"lines": mgr.logs(lines)}


@app.get("/logs/events")
def get_event_logs(lines: int = 50, key: Optional[str] = None):
    cfg = _load_config(key)
    path = _event_log_path(cfg)
    if not path.exists():
        return {"available": False, "path": str(path)}
    rows = _read_csv_tail(path, lines)
    return {"available": True, "path": str(path), "rows": rows}


def _upload_queue_path(cfg, queue_name: str) -> Path:
    normalized = str(queue_name or 'event').strip().lower()
    if normalized not in {'event', 'wheel'}:
        raise HTTPException(status_code=400, detail='queue 必须为 event 或 wheel')
    filename = 'upload_queue.db' if normalized == 'event' else 'wheel_photo_queue.db'
    return _event_output_dir(cfg) / filename


@app.get("/uploads/dead_letters")
def get_upload_dead_letters(
    key: Optional[str] = None,
    queue: str = 'event',
    limit: int = 100,
):
    cfg = _load_config(key)
    path = _upload_queue_path(cfg, queue)
    if not path.exists():
        return {'available': False, 'path': str(path), 'count': 0, 'items': []}
    db = SQLiteUploadQueue(path)
    try:
        return {
            'available': True,
            'path': str(path),
            'count': db.dead_letter_count(),
            'items': db.dead_letters(limit),
        }
    finally:
        db.close()


@app.post("/uploads/dead_letters/retry")
def retry_upload_dead_letter(
    key: Optional[str] = None,
    queue: str = 'event',
    dead_letter_id: Optional[int] = None,
    group_key: str = '',
):
    cfg = _load_config(key)
    path = _upload_queue_path(cfg, queue)
    if not path.exists():
        raise HTTPException(status_code=404, detail='上传队列不存在')
    db = SQLiteUploadQueue(path)
    try:
        if str(group_key or '').strip():
            retried = db.retry_dead_letter_group(group_key)
        elif dead_letter_id is not None:
            retried = 1 if db.retry_dead_letter(dead_letter_id) else 0
        else:
            raise HTTPException(status_code=400, detail='需要 dead_letter_id 或 group_key')
        if retried <= 0:
            raise HTTPException(status_code=404, detail='未找到可重试记录；业务事件请按event ID整组重试')
        return {'status': 'queued', 'retried': retried}
    finally:
        db.close()


@app.get("/logs/detections")
def get_detection_logs(lines: int = 50, key: Optional[str] = None):
    cfg = _load_config(key)
    csv_path = _detection_csv_path(cfg)
    if not csv_path or not csv_path.exists():
        return {"available": False, "path": str(csv_path) if csv_path else ""}
    rows = _read_csv_tail(csv_path, lines)
    return {"available": True, "path": str(csv_path), "rows": rows}


@app.get("/logs/files")
def list_inference_log_files():
    base = _inference_log_dir()
    if not base.exists():
        return {"files": []}
    items = []
    for p in sorted(base.glob("*.log")):
        try:
            st = p.stat()
        except OSError:
            continue
        items.append(
            {
                "name": p.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
            }
        )
    return {"files": items}


@app.get("/logs/file/{name}")
def read_inference_log_file(name: str, lines: int = 400):
    if any(sep in name for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail="文件名非法")
    base = _inference_log_dir()
    path = (base / name).resolve()
    try:
        base_resolved = base.resolve()
    except Exception:
        base_resolved = base
    if not str(path).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="路径越界")
    if not path.exists():
        raise HTTPException(status_code=404, detail="日志文件不存在")
    max_lines = max(1, min(2000, int(lines)))
    buf = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                buf.append(line.rstrip("\n"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取失败: {exc}") from exc
    content = "\n".join(buf)
    return PlainTextResponse(content)


@app.get("/logs", response_class=HTMLResponse)
def logs_page():
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>CleaningCar 日志监控</title>
  <style>
    body { font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 0; background: #111; color: #eee; }
    header { padding: 12px 20px; background: #191d23; border-bottom: 1px solid #242931; }
    main { padding: 16px 20px 40px; display: grid; grid-template-columns: 260px 1fr; gap: 16px; }
    h1 { margin: 0; font-size: 18px; }
    .sidebar { background: #1c2129; border: 1px solid #2a313d; border-radius: 8px; padding: 10px; }
    .content { background: #0f1115; border: 1px solid #2a313d; border-radius: 8px; padding: 10px; }
    ul { list-style: none; padding: 0; margin: 0; max-height: 70vh; overflow-y: auto; }
    li { padding: 6px 8px; cursor: pointer; border-radius: 4px; font-size: 13px; }
    li:hover { background: #2a313d; }
    li.active { background: #345; }
    .meta { font-size: 11px; color: #9aa3b5; }
    pre { white-space: pre-wrap; word-break: break-all; font-size: 12px; line-height: 1.4; }
    .toolbar { margin-bottom: 8px; font-size: 13px; display: flex; gap: 8px; align-items: center; }
    input { background: #000; border-radius: 4px; border: 1px solid #2a313d; color: #eee; padding: 4px 6px; width: 80px; }
    button { border: none; border-radius: 4px; padding: 4px 10px; background: #2f7cf8; color: #fff; cursor: pointer; font-size: 13px; }
    button.secondary { background: #3b3f45; }
    a { color: #8ab4ff; text-decoration: none; }
  </style>
</head>
<body>
  <header>
    <h1>CleaningCar 日志监控</h1>
  </header>
  <main>
    <section class="sidebar">
      <div class="toolbar">
        <span>推理日志文件</span>
        <button class="secondary" onclick="loadFiles()">刷新</button>
      </div>
      <ul id="file-list"></ul>
    </section>
    <section class="content">
      <div class="toolbar">
        <span id="current-file">未选择文件</span>
        <span style="flex:1"></span>
        <label>尾部行数 <input id="line-count" type="number" min="50" max="2000" value="400"></label>
        <button onclick="reloadContent()">刷新内容</button>
        <a href="/" style="margin-left:8px;">返回配置控制台</a>
      </div>
      <pre id="log-content">选择左侧日志文件查看内容。</pre>
    </section>
  </main>
  <script>
    let current = null;
    async function loadFiles() {
      const ul = document.getElementById('file-list');
      ul.innerHTML = '<li>加载中…</li>';
      try {
        const res = await fetch('/logs/files');
        if (!res.ok) throw new Error('请求失败');
        const data = await res.json();
        const files = data.files || [];
        if (!files.length) {
          ul.innerHTML = '<li>暂无日志文件</li>';
          return;
        }
        ul.innerHTML = '';
        files.sort((a, b) => b.mtime - a.mtime);
        for (const f of files) {
          const li = document.createElement('li');
          li.textContent = f.name;
          li.onclick = () => selectFile(f.name, li);
          const meta = document.createElement('div');
          meta.className = 'meta';
          const date = new Date(f.mtime * 1000);
          meta.textContent = date.toLocaleString() + ' · ' + f.size + ' bytes';
          li.appendChild(meta);
          ul.appendChild(li);
        }
      } catch (err) {
        ul.innerHTML = '<li>加载失败: ' + err.message + '</li>';
      }
    }
    async function selectFile(name, li) {
      current = name;
      document.getElementById('current-file').textContent = name;
      const items = document.querySelectorAll('#file-list li');
      items.forEach(x => x.classList.remove('active'));
      li.classList.add('active');
      await reloadContent();
    }
    async function reloadContent() {
      const pre = document.getElementById('log-content');
      if (!current) {
        pre.textContent = '请选择日志文件。';
        return;
      }
      const n = document.getElementById('line-count').value || '400';
      pre.textContent = '加载中…';
      try {
        const res = await fetch('/logs/file/' + encodeURIComponent(current) + '?lines=' + encodeURIComponent(n));
        if (!res.ok) throw new Error('请求失败');
        const text = await res.text();
        pre.textContent = text || '(空文件)';
      } catch (err) {
        pre.textContent = '读取失败: ' + err.message;
      }
    }
    loadFiles();
  </script>
</body>
</html>
    """
    return HTMLResponse(html)


def parse_args():
    parser = argparse.ArgumentParser(description='CleaningCar FastAPI server')
    parser.add_argument('--config', default=str(state.CONFIG_PATH), help='config path')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', help='enable uvicorn reload')
    return parser.parse_args()


def main():
    import uvicorn

    args = parse_args()
    state.set_config_path(Path(args.config).resolve())
    uvicorn.run(
        'web.server:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
        ws='none',
        log_level='warning',
        access_log=False,
    )


if __name__ == '__main__':
    main()
