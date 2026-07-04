#!/usr/bin/env python3
"""
RGA memory bug probe for RK3588-class boards.

This intentionally exposes a few risky combinations that are often involved in
RGA crashes or bad-address style faults:
- virtual-address wrapping vs imported handles
- raw stride vs 16-byte aligned stride for BGR888
- concurrent calls without serialization
 - scheduler core fixed vs auto

Use this on a non-production board first.
"""

import argparse
import ctypes
import os
import random
import threading
import time

import numpy as np

RK_FORMAT_BGR_888 = 0x7 << 8
IM_INTER_LINEAR = 1
IM_STATUS_SUCCESS = 1
IM_CONFIG_SCHEDULER_CORE = 0

RGA_VENDOR = 0
RGA_VERSION = 1

IM_SCHEDULER_RGA3_CORE0 = 1 << 0
IM_SCHEDULER_RGA3_CORE1 = 1 << 1
IM_SCHEDULER_RGA2_CORE0 = 1 << 2
IM_SCHEDULER_RGA2_CORE1 = 1 << 3

CORE_NAME_TO_VALUE = {
    "auto": 0,
    "rga3-0": IM_SCHEDULER_RGA3_CORE0,
    "rga3-1": IM_SCHEDULER_RGA3_CORE1,
    "rga2-0": IM_SCHEDULER_RGA2_CORE0,
    "rga2-1": IM_SCHEDULER_RGA2_CORE1,
}


class _ImColorKeyRange(ctypes.Structure):
    _fields_ = [
        ("max", ctypes.c_int),
        ("min", ctypes.c_int),
    ]


class _ImNn(ctypes.Structure):
    _fields_ = [
        ("scale_r", ctypes.c_int),
        ("scale_g", ctypes.c_int),
        ("scale_b", ctypes.c_int),
        ("offset_r", ctypes.c_int),
        ("offset_g", ctypes.c_int),
        ("offset_b", ctypes.c_int),
    ]


class _RgaBuffer(ctypes.Structure):
    _fields_ = [
        ("vir_addr", ctypes.c_void_p),
        ("phy_addr", ctypes.c_void_p),
        ("fd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("wstride", ctypes.c_int),
        ("hstride", ctypes.c_int),
        ("format", ctypes.c_int),
        ("color_space_mode", ctypes.c_int),
        ("global_alpha", ctypes.c_int),
        ("rd_mode", ctypes.c_int),
        ("color", ctypes.c_int),
        ("colorkey_range", _ImColorKeyRange),
        ("nn", _ImNn),
        ("rop_code", ctypes.c_int),
        ("handle", ctypes.c_uint32),
    ]


class _ImHandleParam(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
    ]


def _align_up(value, align):
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def _iter_librga_candidates():
    env_so = str(os.environ.get("CLEANINGCAR_RGA_SO", "") or "").strip()
    if env_so:
        yield os.path.abspath(env_so)
    yield "/lib/aarch64-linux-gnu/librga.so.2"
    yield "/lib/aarch64-linux-gnu/librga.so"
    yield "/usr/lib/aarch64-linux-gnu/librga.so.2"
    yield "/usr/lib/aarch64-linux-gnu/librga.so"
    yield "librga.so.2"
    yield "librga.so"


class RgaApi:
    def __init__(self):
        self.lib = None
        self.so_path = ""
        self.wrapbuffer_virtualaddr_t = None
        self.wrapbuffer_handle_t = None
        self.importbuffer_virtualaddr = None
        self.releasebuffer_handle = None
        self.imresize_t = None
        self.imStrError_t = None
        self.imconfig = None
        self.querystring = None
        self._load()

    def _load(self):
        last_error = None
        for so_path in _iter_librga_candidates():
            try:
                if os.path.isabs(so_path) and not os.path.exists(so_path):
                    continue
                lib = ctypes.CDLL(so_path)
                wrapbuffer_virtualaddr_t = lib.wrapbuffer_virtualaddr_t
                wrapbuffer_virtualaddr_t.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                wrapbuffer_virtualaddr_t.restype = _RgaBuffer

                wrapbuffer_handle_t = lib.wrapbuffer_handle_t
                wrapbuffer_handle_t.argtypes = [
                    ctypes.c_uint32,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                wrapbuffer_handle_t.restype = _RgaBuffer

                importbuffer_virtualaddr = getattr(lib, "importbuffer_virtualaddr")
                importbuffer_virtualaddr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ImHandleParam)]
                importbuffer_virtualaddr.restype = ctypes.c_uint32

                releasebuffer_handle = getattr(lib, "releasebuffer_handle")
                releasebuffer_handle.argtypes = [ctypes.c_uint32]
                releasebuffer_handle.restype = ctypes.c_int

                imresize_t = lib.imresize_t
                imresize_t.argtypes = [
                    _RgaBuffer,
                    _RgaBuffer,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                imresize_t.restype = ctypes.c_int

                imStrError_t = getattr(lib, "imStrError_t", None)
                if imStrError_t is not None:
                    imStrError_t.argtypes = [ctypes.c_int]
                    imStrError_t.restype = ctypes.c_char_p

                imconfig = getattr(lib, "imconfig", None)
                if imconfig is not None:
                    imconfig.argtypes = [ctypes.c_int, ctypes.c_uint64]
                    imconfig.restype = ctypes.c_int

                querystring = getattr(lib, "querystring", None)
                if querystring is not None:
                    querystring.argtypes = [ctypes.c_int]
                    querystring.restype = ctypes.c_char_p

                self.lib = lib
                self.so_path = so_path
                self.wrapbuffer_virtualaddr_t = wrapbuffer_virtualaddr_t
                self.wrapbuffer_handle_t = wrapbuffer_handle_t
                self.importbuffer_virtualaddr = importbuffer_virtualaddr
                self.releasebuffer_handle = releasebuffer_handle
                self.imresize_t = imresize_t
                self.imStrError_t = imStrError_t
                self.imconfig = imconfig
                self.querystring = querystring
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"failed to load librga: {last_error}")

    def err_text(self, code):
        if self.imStrError_t is None:
            return ""
        try:
            raw = self.imStrError_t(int(code))
            if not raw:
                return ""
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def info_text(self, key):
        if self.querystring is None:
            return ""
        try:
            raw = self.querystring(int(key))
            if not raw:
                return ""
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def set_scheduler_core(self, core_value):
        if not core_value or self.imconfig is None:
            return 0
        return int(self.imconfig(IM_CONFIG_SCHEDULER_CORE, int(core_value)))

    @staticmethod
    def _handle_valid(handle):
        value = int(handle or 0)
        return value not in (0, 0xFFFFFFFF)

    def resize_bgr(
        self,
        src_backing,
        src_w,
        src_h,
        src_wstride,
        dst_backing,
        dst_w,
        dst_h,
        dst_wstride,
        mode,
        core_value=0,
    ):
        if core_value:
            self.set_scheduler_core(core_value)

        if mode == "handle":
            src_param = _ImHandleParam(width=int(src_wstride), height=int(src_h), format=int(RK_FORMAT_BGR_888))
            dst_param = _ImHandleParam(width=int(dst_wstride), height=int(dst_h), format=int(RK_FORMAT_BGR_888))
            src_handle = self.importbuffer_virtualaddr(
                ctypes.c_void_p(int(src_backing.ctypes.data)),
                ctypes.byref(src_param),
            )
            dst_handle = self.importbuffer_virtualaddr(
                ctypes.c_void_p(int(dst_backing.ctypes.data)),
                ctypes.byref(dst_param),
            )
            if not self._handle_valid(src_handle) or not self._handle_valid(dst_handle):
                if self._handle_valid(src_handle):
                    self.releasebuffer_handle(int(src_handle))
                if self._handle_valid(dst_handle):
                    self.releasebuffer_handle(int(dst_handle))
                return 0
            try:
                src_buf = self.wrapbuffer_handle_t(
                    int(src_handle),
                    int(src_w),
                    int(src_h),
                    int(src_wstride),
                    int(src_h),
                    int(RK_FORMAT_BGR_888),
                )
                dst_buf = self.wrapbuffer_handle_t(
                    int(dst_handle),
                    int(dst_w),
                    int(dst_h),
                    int(dst_wstride),
                    int(dst_h),
                    int(RK_FORMAT_BGR_888),
                )
                return int(self.imresize_t(src_buf, dst_buf, 0.0, 0.0, int(IM_INTER_LINEAR), 1))
            finally:
                self.releasebuffer_handle(int(src_handle))
                self.releasebuffer_handle(int(dst_handle))

        src_buf = self.wrapbuffer_virtualaddr_t(
            ctypes.c_void_p(int(src_backing.ctypes.data)),
            int(src_w),
            int(src_h),
            int(src_wstride),
            int(src_h),
            int(RK_FORMAT_BGR_888),
        )
        dst_buf = self.wrapbuffer_virtualaddr_t(
            ctypes.c_void_p(int(dst_backing.ctypes.data)),
            int(dst_w),
            int(dst_h),
            int(dst_wstride),
            int(dst_h),
            int(RK_FORMAT_BGR_888),
        )
        return int(self.imresize_t(src_buf, dst_buf, 0.0, 0.0, int(IM_INTER_LINEAR), 1))


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.ok = 0
        self.fail = 0
        self.exc = 0
        self.first_error = None
        self.started_at = time.time()

    def record_ok(self):
        with self.lock:
            self.calls += 1
            self.ok += 1

    def record_fail(self, message):
        with self.lock:
            self.calls += 1
            self.fail += 1
            if self.first_error is None:
                self.first_error = message

    def record_exc(self, message):
        with self.lock:
            self.calls += 1
            self.exc += 1
            if self.first_error is None:
                self.first_error = message

    def snapshot(self):
        with self.lock:
            elapsed = max(time.time() - self.started_at, 1e-6)
            return {
                "calls": self.calls,
                "ok": self.ok,
                "fail": self.fail,
                "exc": self.exc,
                "elapsed": elapsed,
                "ops_per_sec": self.calls / elapsed,
                "first_error": self.first_error,
            }


def _resolve_profile(args):
    profile = args.profile
    mode = args.mode or ("handle" if profile == "safe" else ("handle" if profile == "stress" else "virtual"))
    core = args.core or ("rga3-0" if profile == "safe" else "auto")
    stride_mode = args.stride_mode or ("align16" if profile == "safe" else ("mixed" if profile == "stress" else "raw"))
    serialize = bool(args.serialize or profile == "safe")
    threads = int(args.threads or (2 if profile == "safe" else (4 if profile == "stress" else 8)))
    return mode, core, stride_mode, serialize, max(1, threads)


def _pick_dims(rng, profile, min_dim, max_dim, dst_width, dst_height):
    if profile == "safe":
        candidates = [
            (960, 720),
            (1280, 720),
            (1280, 960),
            (1600, 900),
        ]
    elif profile == "stress":
        candidates = [
            (641, 641),
            (847, 847),
            (959, 721),
            (1279, 719),
            (1281, 721),
            (1599, 899),
            (511, 511),
            (639, 639),
        ]
    else:
        candidates = [
            (639, 639),
            (641, 641),
            (799, 799),
            (1023, 1023),
            (1279, 719),
            (1281, 721),
            (1919, 1079),
        ]

    usable = []
    for width, height in candidates:
        width = max(int(min_dim), min(int(max_dim), int(width)))
        height = max(int(min_dim), min(int(max_dim), int(height)))
        same_direction = (
            (width >= dst_width and height >= dst_height)
            or (width <= dst_width and height <= dst_height)
        )
        if same_direction:
            usable.append((width, height))
    if not usable:
        usable.append((max(dst_width, min_dim), max(dst_height, min_dim)))
    return rng.choice(usable)


def _choose_stride_mode(rng, requested):
    if requested != "mixed":
        return requested
    return "align16" if rng.random() < 0.5 else "raw"


def _make_bgr_backing(width, height, stride_mode, fill_random, rng):
    if stride_mode == "align16":
        wstride = _align_up(width, 16)
    else:
        wstride = int(width)
    backing = np.empty((int(height), int(wstride), 3), dtype=np.uint8)
    if fill_random:
        backing[:] = rng.integers(0, 256, size=backing.shape, dtype=np.uint8)
    else:
        backing.fill(0)
    return backing, int(wstride)


def _parse_args():
    ap = argparse.ArgumentParser(description="Stress probe for RK3588 librga memory/stride/concurrency bugs.")
    ap.add_argument("--profile", choices=("safe", "stress", "danger"), default="stress")
    ap.add_argument("--mode", choices=("handle", "virtual"), default=None)
    ap.add_argument("--core", choices=tuple(CORE_NAME_TO_VALUE.keys()), default=None)
    ap.add_argument("--stride-mode", choices=("align16", "raw", "mixed"), default=None)
    ap.add_argument("--serialize", action="store_true", help="Serialize all RGA calls with one process lock.")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--iterations-per-thread", type=int, default=5000)
    ap.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means use iterations-per-thread.")
    ap.add_argument("--min-dim", type=int, default=128)
    ap.add_argument("--max-dim", type=int, default=1920)
    ap.add_argument("--dst-width", type=int, default=640)
    ap.add_argument("--dst-height", type=int, default=640)
    ap.add_argument("--fill-random", action="store_true", help="Fill source/dst with random bytes.")
    ap.add_argument("--sleep-ms", type=float, default=0.0)
    ap.add_argument("--log-interval", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=12345)
    return ap.parse_args()


def main():
    args = _parse_args()
    mode, core_name, stride_mode, serialize, threads = _resolve_profile(args)
    core_value = CORE_NAME_TO_VALUE[core_name]
    fill_random = bool(args.fill_random)
    dst_width = max(16, int(args.dst_width))
    dst_height = max(16, int(args.dst_height))
    sleep_seconds = max(0.0, float(args.sleep_ms) / 1000.0)
    duration = max(0.0, float(args.duration))
    iterations_per_thread = max(1, int(args.iterations_per_thread))
    log_interval = max(0.2, float(args.log_interval))
    min_dim = max(16, int(args.min_dim))
    max_dim = max(min_dim, int(args.max_dim))

    api = RgaApi()
    vendor = api.info_text(RGA_VENDOR)
    version = api.info_text(RGA_VERSION)
    print(f"[probe] librga={api.so_path}")
    if vendor:
        print(f"[probe] vendor={vendor}")
    if version:
        print(f"[probe] version={version}")
    print(
        f"[probe] profile={args.profile} mode={mode} core={core_name} stride_mode={stride_mode} "
        f"serialize={serialize} threads={threads} dst={dst_width}x{dst_height} "
        f"iterations_per_thread={iterations_per_thread} duration={duration:.1f}s"
    )
    if args.profile == "danger":
        print("[probe] WARNING: danger profile may hang or reboot an unstable board.")

    global_lock = threading.Lock()
    stats = Stats()
    stop_event = threading.Event()
    deadline = time.time() + duration if duration > 0 else None

    def worker_main(worker_idx):
        rng = random.Random(args.seed + worker_idx)
        np_rng = np.random.default_rng(args.seed + worker_idx)
        loops = 0
        while not stop_event.is_set():
            if deadline is not None and time.time() >= deadline:
                break
            if deadline is None and loops >= iterations_per_thread:
                break
            loops += 1
            try:
                src_w, src_h = _pick_dims(rng, args.profile, min_dim, max_dim, dst_width, dst_height)
                src_stride_mode = _choose_stride_mode(rng, stride_mode)
                dst_stride_mode = _choose_stride_mode(rng, stride_mode)
                src_backing, src_wstride = _make_bgr_backing(src_w, src_h, src_stride_mode, fill_random, np_rng)
                dst_backing, dst_wstride = _make_bgr_backing(
                    dst_width,
                    dst_height,
                    dst_stride_mode,
                    fill_random,
                    np_rng,
                )

                if serialize:
                    with global_lock:
                        ret = api.resize_bgr(
                            src_backing,
                            src_w,
                            src_h,
                            src_wstride,
                            dst_backing,
                            dst_width,
                            dst_height,
                            dst_wstride,
                            mode=mode,
                            core_value=core_value,
                        )
                else:
                    ret = api.resize_bgr(
                        src_backing,
                        src_w,
                        src_h,
                        src_wstride,
                        dst_backing,
                        dst_width,
                        dst_height,
                        dst_wstride,
                        mode=mode,
                        core_value=core_value,
                    )

                if ret < IM_STATUS_SUCCESS:
                    stats.record_fail(
                        f"ret={ret} err={api.err_text(ret)} "
                        f"src={src_w}x{src_h}/stride{src_wstride} "
                        f"dst={dst_width}x{dst_height}/stride{dst_wstride} "
                        f"mode={mode} core={core_name}"
                    )
                else:
                    # Touch the output so the CPU really reads the resulting buffer.
                    _ = int(dst_backing[0, 0, 0]) + int(dst_backing[-1, min(dst_width - 1, dst_wstride - 1), 2])
                    stats.record_ok()
            except Exception as exc:
                stats.record_exc(f"worker={worker_idx} exc={exc}")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    threads_list = [threading.Thread(target=worker_main, args=(i,), daemon=True) for i in range(threads)]
    for thread in threads_list:
        thread.start()

    try:
        while any(thread.is_alive() for thread in threads_list):
            snap = stats.snapshot()
            print(
                f"[probe] calls={snap['calls']} ok={snap['ok']} fail={snap['fail']} "
                f"exc={snap['exc']} ops/s={snap['ops_per_sec']:.1f}"
            )
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(log_interval)
    except KeyboardInterrupt:
        print("[probe] interrupted by user")
    finally:
        stop_event.set()
        for thread in threads_list:
            thread.join(timeout=1.0)

    snap = stats.snapshot()
    print(
        f"[probe:final] calls={snap['calls']} ok={snap['ok']} fail={snap['fail']} "
        f"exc={snap['exc']} elapsed={snap['elapsed']:.1f}s ops/s={snap['ops_per_sec']:.1f}"
    )
    if snap["first_error"]:
        print(f"[probe:final] first_error={snap['first_error']}")

    if snap["fail"] > 0 or snap["exc"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
