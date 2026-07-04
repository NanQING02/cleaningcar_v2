#!/usr/bin/env python3
"""车轮模型 NPU 极限检测脚本（板端专用）

同时拉左右车轮 RTSP，用 RKNN 车轮模型在 NPU 上「拉满」推理（不限 target_fps，
每来一帧就推），可视化保存抽帧标注图，并统计各档位检出次数、推理耗时、丢帧率。

只能在 RK3588 板端（rknnlite 可用）运行；PC 端跑会因为 import rknnlite 失败而退出。

输出目录（默认 wheel_smoke_output/<时间戳>/）：
  resolved_settings.json   本次配置快照
  annotated/<side>_<frame>.jpg   抽帧的可视化标注图
  detections.jsonl         每帧检出明细（side/frame_idx/ts/detections[]）
  errors.jsonl             推理异常流水
  summary.json             汇总：解码/推理帧数、各档位计数、avg/p95 推理耗时

典型用法（板端项目根目录）：
  python tools/wheel_model_smoke.py --duration 120 --core-mask 0:1:2
  python tools/wheel_model_smoke.py --left-source rtsp://... --right-source rtsp://... \
      --model models/wheel/2026.4.28CRwheel.rknn --conf-thresh 0.25 --save-every 10
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np

def _resolve_project_root() -> Path:
    """优先用 cwd（若其中存在 cleaningcar 包），其次用 __file__/..，最后退到 __file__/..。"""
    candidates = []
    here = Path(__file__).resolve().parent
    candidates.append(here.parent)
    candidates.append(Path.cwd())
    for cand in candidates:
        if (cand / "cleaningcar").is_dir() and (cand / "configs").is_dir():
            return cand.resolve()
    return here.parent.resolve()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleaningcar.fp_detect import FpModelPostprocessor
from cleaningcar.video_io import create_video_reader, parse_core_mask

DEFAULT_LEFT_SOURCE = ""
DEFAULT_RIGHT_SOURCE = ""
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "wheel" / "2026.4.28CRwheel.rknn")
DEFAULT_CLASSES = ["0-25", "25-50", "50-75", "75-100"]
WHEEL_SIDES = ("left", "right")

CLASS_COLORS = {
    0: (80, 200, 80),
    1: (40, 180, 240),
    2: (40, 120, 240),
    3: (40, 40, 220),
}


def class_name(classes, cls_id):
    idx = int(cls_id)
    if 0 <= idx < len(classes):
        return str(classes[idx])
    return str(idx)


class LatestFrameSlot:
    """单槽最新帧缓冲：reader put，infer 取最新。"""

    def __init__(self):
        self._cond = threading.Condition()
        self._seq = 0
        self._item = None

    def put(self, item):
        with self._cond:
            self._seq += 1
            self._item = item
            self._cond.notify_all()

    def peek(self):
        with self._cond:
            return self._seq, self._item

    def wait_for_update(self, last_seq, timeout=0.2):
        deadline = time.time() + max(0.0, float(timeout))
        with self._cond:
            while self._seq <= last_seq:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cond.wait(timeout=remaining)
            return self._seq, self._item


class WheelBenchReader(threading.Thread):
    def __init__(self, side, source, hw_decode, config, frame_slot, stop_event, reconnect_delay=2.0):
        super().__init__(daemon=True, name=f"wheel-smoke-reader-{side}")
        self.side = side
        self.source = source
        self.reader_args = SimpleNamespace(hw_decode=bool(hw_decode), _config=config)
        self.frame_slot = frame_slot
        self.stop_event = stop_event
        self.reconnect_delay = float(reconnect_delay)
        self.frames = 0
        self.open_failures = 0
        self.last_open_meta: Dict[str, str] = {}

    def run(self):
        cap = None
        while not self.stop_event.is_set():
            if cap is None or not getattr(cap, "isOpened", lambda: False)():
                cap, meta = create_video_reader(self.source, self.reader_args)
                if cap is None or not cap.isOpened():
                    self.open_failures += 1
                    print(
                        f"[{self.side}] reader open failed (#{self.open_failures}) source={self.source}",
                        flush=True,
                    )
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    cap = None
                    if self.stop_event.wait(self.reconnect_delay):
                        break
                    continue
                self.last_open_meta = dict(meta or {})
                print(
                    f"[{self.side}] reader opened mode={meta.get('decode_mode')} "
                    f"backend={meta.get('decode_backend')} fallback={meta.get('fallback_used')}",
                    flush=True,
                )
            ok, frame = cap.read()
            if not ok or frame is None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                if self.stop_event.wait(self.reconnect_delay):
                    break
                continue
            self.frames += 1
            self.frame_slot.put((frame, time.time()))
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


class WheelBenchInfer(threading.Thread):
    def __init__(
        self,
        side,
        model_path,
        classes,
        imgsz,
        conf_thresh,
        nms_thresh,
        core_mask,
        frame_slot,
        stop_event,
        bench,
        save_every=15,
        image_quality=85,
    ):
        super().__init__(daemon=True, name=f"wheel-smoke-infer-{side}")
        self.side = side
        self.model_path = Path(model_path)
        self.classes = list(classes)
        self.imgsz = int(imgsz)
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.core_mask = core_mask
        self.frame_slot = frame_slot
        self.stop_event = stop_event
        self.bench = bench
        self.save_every = max(1, int(save_every))
        self.image_quality = int(min(max(int(image_quality), 1), 100))
        self.frames = 0
        self.detections = 0
        self.infer_time = 0.0
        self._output_mode = "6"
        self._postprocessor: Optional[FpModelPostprocessor] = None

    def _build_runtime(self):
        from rknnlite.api import RKNNLite

        return RKNNLite()

    def _build_postprocessor(self, output_mode):
        return FpModelPostprocessor(
            img_size=(self.imgsz, self.imgsz),
            obj_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            output_mode=output_mode,
            num_classes=len(self.classes),
        )

    def _ensure_output_mode(self, outputs):
        expected = "9" if len(outputs) == 9 else "6"
        if expected == self._output_mode and self._postprocessor is not None:
            return
        self._output_mode = expected
        self._postprocessor = self._build_postprocessor(expected)
        print(f"[{self.side}] postprocess mode={self._postprocessor.describe_mode()}", flush=True)

    def _annotate(self, frame, boxes, classes, scores):
        annotated = frame.copy()
        for box, cls_id, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            color = CLASS_COLORS.get(int(cls_id), (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name(self.classes, int(cls_id))}:{float(score):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(th + 1, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"[{self.side}] frame={self.frames} dets={len(boxes)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def _maybe_save_annotated(self, frame, boxes, classes, scores):
        if self.frames % self.save_every != 0:
            return
        out_path = self.bench.annotated_dir / f"{self.side}_{self.frames:06d}.jpg"
        try:
            cv2.imwrite(
                str(out_path),
                self._annotate(frame, boxes, classes, scores),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality],
            )
        except Exception:
            pass

    def run(self):
        if not self.model_path.exists():
            print(f"[{self.side}] model missing: {self.model_path}", flush=True)
            return
        try:
            rk = self._build_runtime()
            if rk.load_rknn(str(self.model_path)) != 0:
                raise RuntimeError("load_rknn failed")
            init_kwargs = {}
            if self.core_mask:
                init_kwargs["core_mask"] = self.core_mask
            if rk.init_runtime(**init_kwargs) != 0:
                raise RuntimeError("init_runtime failed")
        except Exception as exc:
            import traceback

            print(f"[{self.side}] runtime init failed: {exc}", flush=True)
            traceback.print_exc()
            try:
                rk.release()
            except Exception:
                pass
            return

        self._postprocessor = self._build_postprocessor(self._output_mode)
        print(
            f"[{self.side}] infer ready imgsz={self.imgsz} conf={self.conf_thresh} "
            f"nms={self.nms_thresh} core_mask={self.core_mask}",
            flush=True,
        )

        last_seq = 0
        try:
            while not self.stop_event.is_set():
                seq, item = self.frame_slot.peek()
                if item is None or seq == last_seq:
                    seq, item = self.frame_slot.wait_for_update(last_seq, timeout=0.2)
                    if item is None or seq == last_seq:
                        continue
                last_seq = seq
                frame, capture_ts = item
                started = time.time()
                try:
                    img_input, lb_info = self._postprocessor.prepare(frame)
                    outputs = rk.inference(inputs=[img_input], data_format=["nhwc"])
                    if not outputs:
                        continue
                    self._ensure_output_mode(outputs)
                    boxes, classes, scores = self._postprocessor.postprocess(outputs)
                    if boxes is not None and classes is not None and scores is not None:
                        boxes = self._postprocessor.map_boxes_to_original(boxes, lb_info)
                        self.detections += int(len(boxes))
                        self.bench.record_detections(self.side, self.frames, capture_ts, boxes, classes, scores, self.classes)
                        self._maybe_save_annotated(frame, boxes, classes, scores)
                    else:
                        self._maybe_save_annotated(frame, [], [], [])
                except Exception as exc:
                    self.bench.record_infer_error(self.side, str(exc))
                finally:
                    elapsed = max(0.0, time.time() - started)
                    self.infer_time += elapsed
                    self.bench.record_infer_timing(self.side, elapsed)
                    self.frames += 1
        finally:
            try:
                rk.release()
            except Exception:
                pass


class WheelBench:
    def __init__(self, output_dir, retain_score_topk=8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir = self.output_dir / "annotated"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.detections_path = self.output_dir / "detections.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.retain_score_topk = int(retain_score_topk)
        self._lock = threading.Lock()
        self._class_counts = {side: defaultdict(int) for side in WHEEL_SIDES}
        self._score_samples = {side: defaultdict(list) for side in WHEEL_SIDES}
        self._infer_times = {side: deque(maxlen=300) for side in WHEEL_SIDES}
        self._infer_errors = defaultdict(int)
        self._det_file = self.detections_path.open("a", encoding="utf-8")
        self._err_file = self.errors_path.open("a", encoding="utf-8")

    def record_detections(self, side, frame_idx, capture_ts, boxes, classes, scores, class_names):
        ts = time.time()
        with self._lock:
            for cls_id, score in zip(classes, scores):
                cls_key = int(cls_id)
                self._class_counts[side][cls_key] += 1
                samples = self._score_samples[side][cls_key]
                if len(samples) < self.retain_score_topk:
                    samples.append(round(float(score), 4))
                else:
                    idx_min = int(np.argmin(samples))
                    if float(score) > samples[idx_min]:
                        samples[idx_min] = round(float(score), 4)
            det_records = [
                {
                    "classId": int(c),
                    "className": class_name(class_names, int(c)),
                    "score": round(float(s), 4),
                    "box": [int(round(v)) for v in box[:4]],
                }
                for box, c, s in zip(boxes, classes, scores)
            ]
            line = json.dumps(
                {
                    "ts": ts,
                    "side": side,
                    "frame_idx": frame_idx,
                    "capture_ts": capture_ts,
                    "detections": det_records,
                },
                ensure_ascii=False,
            )
            self._det_file.write(line + "\n")
            self._det_file.flush()

    def record_infer_error(self, side, message):
        with self._lock:
            self._infer_errors[side] += 1
            self._err_file.write(
                json.dumps({"ts": time.time(), "side": side, "message": message}, ensure_ascii=False) + "\n"
            )
            self._err_file.flush()

    def record_infer_timing(self, side, elapsed):
        with self._lock:
            self._infer_times[side].append(float(elapsed))

    def snapshot(self, readers, infers, started_ts):
        now = time.time()
        elapsed = max(0.001, now - started_ts)
        snap = {"elapsed": round(elapsed, 2), "sides": {}}
        for side in WHEEL_SIDES:
            reader = readers.get(side)
            infer = infers.get(side)
            times = list(self._infer_times.get(side, []))
            if times:
                avg_ms = sum(times) / len(times) * 1000.0
                p95 = times[int(min(len(times) - 1, len(times) * 0.95))]
                p95_ms = p95 * 1000.0
            else:
                avg_ms = 0.0
                p95_ms = 0.0
            infer_frames = int(getattr(infer, "frames", 0) or 0)
            infer_time_total = float(getattr(infer, "infer_time", 0.0) or 0.0)
            infer_fps = (infer_frames / infer_time_total) if infer_time_total > 0 else 0.0
            decode_frames = int(getattr(reader, "frames", 0) or 0)
            drop_rate = (decode_frames - infer_frames) / max(1, decode_frames)
            class_counts = {
                class_name(DEFAULT_CLASSES, int(k)) if int(k) < len(DEFAULT_CLASSES) else str(int(k)): int(v)
                for k, v in sorted(self._class_counts.get(side, {}).items())
            }
            score_samples = {
                class_name(DEFAULT_CLASSES, int(k)) if int(k) < len(DEFAULT_CLASSES) else str(int(k)): list(v)
                for k, v in sorted(self._score_samples.get(side, {}).items())
            }
            snap["sides"][side] = {
                "decode_frames": decode_frames,
                "decode_fps": round(decode_frames / elapsed, 2),
                "open_failures": int(getattr(reader, "open_failures", 0) or 0),
                "decode_meta": dict(getattr(reader, "last_open_meta", {}) or {}),
                "infer_frames": infer_frames,
                "infer_fps": round(infer_fps, 2),
                "detections": int(getattr(infer, "detections", 0) or 0),
                "infer_avg_ms": round(avg_ms, 2),
                "infer_p95_ms": round(p95_ms, 2),
                "drop_rate": round(drop_rate, 3),
                "class_counts": class_counts,
                "score_samples": score_samples,
                "infer_errors": int(self._infer_errors.get(side, 0)),
            }
        return snap

    def finalize(self, readers, infers, started_ts):
        snap = self.snapshot(readers, infers, started_ts)
        with self._lock:
            try:
                self._det_file.close()
            except Exception:
                pass
            try:
                self._err_file.close()
            except Exception:
                pass
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        return snap


def parse_args():
    parser = argparse.ArgumentParser(description="车轮模型 NPU 极限检测脚本")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.json"), help="配置文件路径")
    parser.add_argument("--left-source", default="", help="覆盖左 RTSP；空=读配置或脚本内置默认")
    parser.add_argument("--right-source", default="", help="覆盖右 RTSP；空=读配置或脚本内置默认")
    parser.add_argument("--model", default="", help="覆盖车轮模型路径")
    parser.add_argument("--duration", type=float, default=60.0, help="运行时长秒，<=0 表示一直跑到 Ctrl+C")
    parser.add_argument("--imgsz", type=int, default=0, help="覆盖 wheel.imgsz；0=读配置或 640")
    parser.add_argument("--conf-thresh", type=float, default=-1.0, help="覆盖 wheel.conf_thresh；<0=读配置")
    parser.add_argument("--nms-thresh", type=float, default=-1.0, help="覆盖 wheel.nms_thresh；<0=读配置")
    parser.add_argument(
        "--core-mask",
        default="all",
        help="NPU core mask；可选 0 / 0,1,2 / all（推荐 all 拉满三核）；parse_core_mask 只认逗号或 all，传 '0:1:2' 会失效；空字符串则交给 RKNN 默认",
    )
    parser.add_argument("--hw-decode", dest="hw_decode", action="store_true", default=True, help="使用硬解码（默认）")
    parser.add_argument("--sw-decode", dest="hw_decode", action="store_false", help="强制软解")
    parser.add_argument("--save-every", type=int, default=15, help="每 N 帧保存一张可视化标注图，默认 15")
    parser.add_argument("--status-interval", type=float, default=5.0, help="状态打印间隔秒")
    parser.add_argument("--output-dir", default="", help="输出目录，默认 wheel_smoke_output/<时间戳>")
    return parser.parse_args()


def load_config(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        print(f"[warn] config not found: {p}, fallback to built-in defaults", flush=True)
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_settings(args, config):
    wheel_cfg = (config or {}).get("wheel", {}) or {}
    left = (args.left_source or wheel_cfg.get("left_source") or DEFAULT_LEFT_SOURCE).strip()
    right = (args.right_source or wheel_cfg.get("right_source") or DEFAULT_RIGHT_SOURCE).strip()
    model_value = (args.model or wheel_cfg.get("model") or DEFAULT_MODEL).strip()
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    imgsz = int(args.imgsz) if args.imgsz > 0 else int(wheel_cfg.get("imgsz", 640) or 640)
    conf = float(args.conf_thresh if args.conf_thresh >= 0 else wheel_cfg.get("conf_thresh", 0.25))
    nms = float(args.nms_thresh if args.nms_thresh >= 0 else wheel_cfg.get("nms_thresh", 0.45))
    core_mask_text = (args.core_mask or "").strip()
    core_mask = parse_core_mask(core_mask_text)
    classes = list(wheel_cfg.get("classes") or DEFAULT_CLASSES)
    return {
        "left": left,
        "right": right,
        "model": str(model_path),
        "imgsz": imgsz,
        "conf": conf,
        "nms": nms,
        "core_mask": core_mask,
        "core_mask_text": core_mask_text or "<rknn-default>",
        "classes": classes,
    }


def preflight_npu(model_path, core_mask, classes, imgsz, conf, nms):
    """启动前在主线程做一次 NPU 模型加载 + 推理自检，把失败前置暴露。"""
    try:
        from rknnlite.api import RKNNLite
    except Exception as exc:
        print(f"[preflight][FAIL] import rknnlite failed: {exc}", flush=True)
        return False
    if not Path(model_path).exists():
        print(f"[preflight][FAIL] model not found: {model_path}", flush=True)
        return False
    rk = RKNNLite()
    try:
        ret = rk.load_rknn(str(model_path))
        if ret != 0:
            print(f"[preflight][FAIL] load_rknn returned {ret} (model={model_path})", flush=True)
            return False
        init_kwargs = {}
        if core_mask:
            init_kwargs["core_mask"] = core_mask
        ret = rk.init_runtime(**init_kwargs)
        if ret != 0:
            print(f"[preflight][FAIL] init_runtime returned {ret} core_mask={core_mask}", flush=True)
            return False
        post = FpModelPostprocessor(
            img_size=(int(imgsz), int(imgsz)),
            obj_thresh=float(conf),
            nms_thresh=float(nms),
            output_mode="6",
            num_classes=len(classes),
        )
        dummy = np.zeros((int(imgsz), int(imgsz), 3), dtype=np.uint8)
        img_input, _ = post.prepare(dummy)
        outputs = rk.inference(inputs=[img_input], data_format=["nhwc"])
        if not outputs:
            print("[preflight][FAIL] inference returned empty outputs", flush=True)
            return False
        print(
            f"[preflight][OK] model={model_path} core_mask={core_mask} "
            f"outputs={len(outputs)} shapes={[getattr(o, 'shape', None) for o in outputs]}",
            flush=True,
        )
        return True
    except Exception as exc:
        import traceback

        print(f"[preflight][FAIL] exception during NPU preflight: {exc}", flush=True)
        traceback.print_exc()
        return False
    finally:
        try:
            rk.release()
        except Exception:
            pass


def main():
    args = parse_args()
    config = load_config(args.config)
    settings = resolve_settings(args, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(PROJECT_ROOT / "wheel_smoke_output" / timestamp)

    bench = WheelBench(output_dir)
    stop_event = threading.Event()

    print(f"[wheel-smoke] output_dir={output_dir}", flush=True)
    print(f"[wheel-smoke] left={settings['left']}", flush=True)
    print(f"[wheel-smoke] right={settings['right']}", flush=True)
    print(
        f"[wheel-smoke] model={settings['model']} imgsz={settings['imgsz']} "
        f"conf={settings['conf']} nms={settings['nms']} core={settings['core_mask_text']} "
        f"hw_decode={args.hw_decode} save_every={args.save_every} duration={args.duration}",
        flush=True,
    )

    if not preflight_npu(
        settings["model"], settings["core_mask"], settings["classes"],
        settings["imgsz"], settings["conf"], settings["nms"],
    ):
        print("[wheel-smoke] NPU preflight failed, exit. Resolve the error above before retrying.", flush=True)
        return

    (bench.output_dir / "resolved_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readers: Dict[str, WheelBenchReader] = {}
    infers: Dict[str, WheelBenchInfer] = {}
    for side, source in (("left", settings["left"]), ("right", settings["right"])):
        if not source:
            continue
        slot = LatestFrameSlot()
        reader = WheelBenchReader(
            side=side,
            source=source,
            hw_decode=args.hw_decode,
            config=config,
            frame_slot=slot,
            stop_event=stop_event,
        )
        infer = WheelBenchInfer(
            side=side,
            model_path=settings["model"],
            classes=settings["classes"],
            imgsz=settings["imgsz"],
            conf_thresh=settings["conf"],
            nms_thresh=settings["nms"],
            core_mask=settings["core_mask"],
            frame_slot=slot,
            stop_event=stop_event,
            bench=bench,
            save_every=args.save_every,
        )
        readers[side] = reader
        infers[side] = infer

    if not readers:
        print("[wheel-smoke] no left/right source configured, exit", flush=True)
        return

    started_ts = time.time()
    for reader in readers.values():
        reader.start()
    for infer in infers.values():
        infer.start()

    def status_loop():
        while not stop_event.is_set():
            snap = bench.snapshot(readers, infers, started_ts)
            for side, info in snap["sides"].items():
                print(
                    f"[status] {side}: decode={info['decode_frames']}({info['decode_fps']}fps) "
                    f"infer={info['infer_frames']}({info['infer_fps']}fps) drop={info['drop_rate']:.2f} "
                    f"dets={info['detections']} avg={info['infer_avg_ms']}ms p95={info['infer_p95_ms']}ms "
                    f"err={info['infer_errors']} classes={info['class_counts']}",
                    flush=True,
                )
            if stop_event.wait(args.status_interval):
                break

    status_thread = threading.Thread(target=status_loop, daemon=True, name="wheel-smoke-status")
    status_thread.start()

    try:
        if args.duration > 0:
            stop_event.wait(args.duration)
        else:
            while not stop_event.is_set():
                stop_event.wait(1.0)
    except KeyboardInterrupt:
        print("[wheel-smoke] interrupted, stopping...", flush=True)
    finally:
        stop_event.set()
        for reader in readers.values():
            reader.join(timeout=2.0)
        for infer in infers.values():
            infer.join(timeout=2.0)
        status_thread.join(timeout=1.0)
        snap = bench.finalize(readers, infers, started_ts)
        print(f"[wheel-smoke] done. summary={bench.summary_path}", flush=True)
        print(json.dumps(snap, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
