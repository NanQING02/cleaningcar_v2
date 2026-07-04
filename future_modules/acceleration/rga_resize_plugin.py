import os
import ctypes
import threading

import cv2
import numpy as np


RK_FORMAT_BGR_888 = 0x7 << 8
IM_INTER_LINEAR = 1
IM_STATUS_SUCCESS = 1


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


def _is_true_env(name):
    value = os.environ.get(name, "")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _read_positive_int_env(name, default):
    raw = os.environ.get(name, "")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    return max(1, value)


def _align_up(value, align):
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def _align_bgr888_stride(width):
    return _align_up(width, 16)


def _iter_librga_candidates():
    env_so = os.environ.get("CLEANINGCAR_RGA_SO", "").strip()
    if env_so:
        yield os.path.abspath(env_so)

    # Prefer official RK3588 system paths.
    yield "/lib/aarch64-linux-gnu/librga.so.2"
    yield "/lib/aarch64-linux-gnu/librga.so"
    yield "/usr/lib/aarch64-linux-gnu/librga.so.2"
    yield "/usr/lib/aarch64-linux-gnu/librga.so"
    yield "/usr/local/lib/librga.so"

    # Fallback to linker-resolved soname.
    yield "librga.so.2"
    yield "librga.so"


def _bind_symbols(lib):
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

    return wrapbuffer_virtualaddr_t, imresize_t, imStrError_t


def _load_librga():
    last_error = None
    for so_path in _iter_librga_candidates():
        try:
            if os.path.isabs(so_path) and not os.path.exists(so_path):
                continue
            lib = ctypes.CDLL(so_path)
            wrapbuffer_virtualaddr_t, imresize_t, imStrError_t = _bind_symbols(lib)
            return lib, so_path, wrapbuffer_virtualaddr_t, imresize_t, imStrError_t, None
        except Exception as exc:
            last_error = exc
    return None, "", None, None, None, last_error


if _is_true_env("CLEANINGCAR_RGA_DISABLE"):
    _lib = None
    _wrapbuffer_virtualaddr_t = None
    _imresize_t = None
    _imStrError_t = None
    _load_error = None
    SO_PATH = ""
    RGA_OK = False
    print("[rga-resize-plugin] disabled by CLEANINGCAR_RGA_DISABLE")
else:
    (
        _lib,
        SO_PATH,
        _wrapbuffer_virtualaddr_t,
        _imresize_t,
        _imStrError_t,
        _load_error,
    ) = _load_librga()
    RGA_OK = _lib is not None
    if RGA_OK:
        print("[rga-resize-plugin] loaded", SO_PATH)
    else:
        if _load_error is None:
            print("[rga-resize-plugin] load so failed: no valid librga found")
        else:
            print("[rga-resize-plugin] load so failed:", _load_error)

_RGA_CALL_FAIL_COUNT = 0
_RGA_SKIP_COUNT = 0
_RGA_CALL_LOCK = threading.Lock()
_RGA_MIN_DIM = _read_positive_int_env("CLEANINGCAR_RGA_MIN_DIM", 64)


def _resize_with_librga(src, dst):
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = dst.shape[:2]
    src_wstride = src.shape[1]
    dst_wstride = dst.shape[1]

    src_buf = _wrapbuffer_virtualaddr_t(
        ctypes.c_void_p(int(src.ctypes.data)),
        int(src_w),
        int(src_h),
        int(src_wstride),
        int(src_h),
        int(RK_FORMAT_BGR_888),
    )
    dst_buf = _wrapbuffer_virtualaddr_t(
        ctypes.c_void_p(int(dst.ctypes.data)),
        int(dst_w),
        int(dst_h),
        int(dst_wstride),
        int(dst_h),
        int(RK_FORMAT_BGR_888),
    )
    return int(_imresize_t(src_buf, dst_buf, 0.0, 0.0, int(IM_INTER_LINEAR), 1))


def _err_to_text(code):
    if _imStrError_t is None:
        return ""
    try:
        raw = _imStrError_t(int(code))
        if not raw:
            return ""
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _log_rga_skip(reason, image, tw, th):
    global _RGA_SKIP_COUNT
    _RGA_SKIP_COUNT += 1
    if _RGA_SKIP_COUNT > 3 and _RGA_SKIP_COUNT % 100 != 0:
        return
    shape = getattr(image, "shape", None)
    dtype = getattr(image, "dtype", None)
    contiguous = bool(getattr(getattr(image, "flags", None), "c_contiguous", False))
    print(
        "[rga-resize-plugin] fallback to cv2: "
        f"reason={reason} src_shape={shape} dst_shape=({th}, {tw}, 3) "
        f"dtype={dtype} contiguous={contiguous} count={_RGA_SKIP_COUNT}"
    )


def _cv_resize(im, tw, th):
    return cv2.resize(im, (max(1, int(tw)), max(1, int(th))), interpolation=cv2.INTER_LINEAR)


def _rga_skip_reason(im, tw, th):
    if im is None:
        return "image_none"
    if not isinstance(im, np.ndarray):
        return "not_ndarray"
    if im.ndim != 3 or im.shape[2] != 3:
        return "invalid_shape"
    if im.dtype != np.uint8:
        return "invalid_dtype"
    if im.size == 0:
        return "empty_image"
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
        return "empty_shape"
    if tw <= 0 or th <= 0:
        return "invalid_target"
    if min(h, w, th, tw) < _RGA_MIN_DIM:
        return f"small_dim_lt_{_RGA_MIN_DIM}"
    return ""


def rga_resize(im, new_unpad):
    tw, th = int(new_unpad[0]), int(new_unpad[1])
    if im is None:
        return None
    h, w = im.shape[:2]
    if w == tw and h == th:
        return im
    if tw <= 0 or th <= 0:
        return _cv_resize(im, tw, th)
    if (not RGA_OK) or (_lib is None):
        return _cv_resize(im, tw, th)
    skip_reason = _rga_skip_reason(im, tw, th)
    if skip_reason:
        _log_rga_skip(skip_reason, im, tw, th)
        return _cv_resize(im, tw, th)

    src_wstride = _align_bgr888_stride(w)
    dst_wstride = _align_bgr888_stride(tw)
    src = np.ascontiguousarray(im)

    if src_wstride != w:
        src_pad = np.zeros((h, src_wstride, 3), dtype=np.uint8)
        src_pad[:, :w, :] = src
        src = src_pad

    dst = np.empty((th, dst_wstride, 3), dtype=np.uint8)
    with _RGA_CALL_LOCK:
        ret = _resize_with_librga(src, dst)
    if ret < IM_STATUS_SUCCESS:
        global _RGA_CALL_FAIL_COUNT
        _RGA_CALL_FAIL_COUNT += 1
        if _RGA_CALL_FAIL_COUNT <= 3 or _RGA_CALL_FAIL_COUNT % 100 == 0:
            err_text = _err_to_text(ret)
            if err_text:
                print(
                    f"[rga-resize-plugin] imresize_t failed: ret={ret} err='{err_text}' "
                    f"src_shape={src.shape} dst_shape={dst.shape} "
                    f"src_stride={src_wstride} dst_stride={dst_wstride} "
                    f"count={_RGA_CALL_FAIL_COUNT}"
                )
            else:
                print(
                    f"[rga-resize-plugin] imresize_t failed: ret={ret} "
                    f"src_shape={src.shape} dst_shape={dst.shape} "
                    f"src_stride={src_wstride} dst_stride={dst_wstride} "
                    f"count={_RGA_CALL_FAIL_COUNT}"
                )
        return _cv_resize(im, tw, th)

    if dst_wstride == tw:
        return dst
    return dst[:, :tw, :].copy()
