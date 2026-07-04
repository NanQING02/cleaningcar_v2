from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - Pillow missing is handled at runtime.
    Image = None
    ImageDraw = None
    ImageFont = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_FONT_PATH = PROJECT_ROOT / 'fonts' / 'platech.ttf'
SYSTEM_FONT_CANDIDATES = (
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf'),
    Path('/usr/share/fonts/truetype/arphic/ukai.ttc'),
    Path('/usr/share/fonts/truetype/arphic/uming.ttc'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    Path('C:/Windows/Fonts/simhei.ttf'),
    Path('C:/Windows/Fonts/msyh.ttc'),
)

_WARNED_MESSAGES = set()
_FONT_SOURCE = None


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(message)


@lru_cache(maxsize=16)
def _resolve_font_path() -> Path | None:
    global _FONT_SOURCE
    if ImageFont is None:
        _warn_once('[text-render] Pillow unavailable, only ASCII cv2.putText fallback will be used.')
        return None
    for candidate in (PROJECT_FONT_PATH, *SYSTEM_FONT_CANDIDATES):
        if not candidate.exists():
            continue
        try:
            ImageFont.truetype(str(candidate), size=24)
        except Exception:
            continue
        _FONT_SOURCE = str(candidate)
        return candidate
    _warn_once(
        '[text-render] no usable Chinese font found; non-ASCII text will degrade to ASCII fallback.'
    )
    return None


@lru_cache(maxsize=128)
def _load_font(size: int):
    font_path = _resolve_font_path()
    if font_path is None or ImageFont is None:
        return None
    try:
        return ImageFont.truetype(str(font_path), size=max(12, int(size)))
    except Exception as exc:
        _warn_once(f'[text-render] failed to load font {font_path}: {exc}')
        return None


def font_source() -> str:
    if _resolve_font_path() is None:
        return ''
    return _FONT_SOURCE or ''


def _font_size_from_scale(font_scale: float, thickness: int) -> int:
    scale = max(0.3, float(font_scale))
    return max(14, int(round(scale * 30.0 + max(0, thickness - 1) * 2.0)))


def _measure_text(text: str, font, outline_width: int) -> Tuple[int, int, int, int]:
    dummy = Image.new('RGB', (1, 1), (0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=outline_width)
    return bbox


def _ascii_fallback(text: str) -> str:
    try:
        return text.encode('ascii', 'replace').decode('ascii')
    except Exception:
        return ''.join(ch if ord(ch) < 128 else '?' for ch in str(text))


def draw_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    *,
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    anchor: str = 'lb',
) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    if not text:
        return frame

    text = str(text)
    font = _load_font(_font_size_from_scale(font_scale, thickness))
    if font is None:
        safe_text = text if text.isascii() else _ascii_fallback(text)
        if safe_text != text:
            _warn_once('[text-render] Chinese font missing, falling back to ASCII-safe labels.')
        cv2.putText(
            frame,
            safe_text,
            (int(org[0]), int(org[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color,
            max(1, int(thickness)),
            cv2.LINE_AA,
        )
        return frame

    outline_width = max(1, int(thickness))
    x = int(org[0])
    y = int(org[1])
    bbox = _measure_text(text, font, outline_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    if text_w <= 0 or text_h <= 0:
        return frame

    if anchor == 'lt':
        left = x
        top = y
    else:
        left = x
        top = y - text_h

    frame_h, frame_w = frame.shape[:2]
    left = max(0, min(left, max(0, frame_w - text_w)))
    top = max(0, min(top, max(0, frame_h - text_h)))

    pad = max(2, outline_width + 1)
    patch_left = max(0, left - pad)
    patch_top = max(0, top - pad)
    patch_right = min(frame_w, left + text_w + pad)
    patch_bottom = min(frame_h, top + text_h + pad)
    if patch_right <= patch_left or patch_bottom <= patch_top:
        return frame

    patch = frame[patch_top:patch_bottom, patch_left:patch_right]
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch_img = Image.fromarray(patch_rgb)
    draw = ImageDraw.Draw(patch_img)

    draw_x = (left - patch_left) - bbox[0]
    draw_y = (top - patch_top) - bbox[1]
    draw.text(
        (draw_x, draw_y),
        text,
        font=font,
        fill=(int(color[2]), int(color[1]), int(color[0])),
        stroke_width=outline_width,
        stroke_fill=(int(outline_color[2]), int(outline_color[1]), int(outline_color[0])),
    )

    frame[patch_top:patch_bottom, patch_left:patch_right] = cv2.cvtColor(
        np.asarray(patch_img),
        cv2.COLOR_RGB2BGR,
    )
    return frame
