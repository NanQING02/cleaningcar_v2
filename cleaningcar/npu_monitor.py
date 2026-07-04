from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


NPU_DEVFREQ_DIR = Path("/sys/class/devfreq/fdab0000.npu")
RKNPU_DEBUG_DIR = Path("/sys/kernel/debug/rknpu")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _read_int(path: Path) -> Optional[int]:
    value = _read_text(path)
    if not value:
        return None
    try:
        return int(value.split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _hz_to_mhz(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def parse_rknpu_core_loads(text: str) -> Dict[int, float]:
    loads: Dict[int, float] = {}
    for match in re.finditer(r"Core\s*(\d+)\s*:\s*([0-9.]+)\s*%", str(text or ""), re.IGNORECASE):
        try:
            loads[int(match.group(1))] = float(match.group(2))
        except (TypeError, ValueError):
            continue
    return loads


def parse_devfreq_load(text: str):
    raw = str(text or "").strip()
    if not raw:
        return None, None
    match = re.search(r"([0-9.]+)\s*@\s*([0-9]+)\s*Hz", raw, re.IGNORECASE)
    if not match:
        return None, None
    try:
        return float(match.group(1)), int(match.group(2))
    except (TypeError, ValueError):
        return None, None


def snapshot_npu_status() -> dict:
    debug_load_raw = _read_text(RKNPU_DEBUG_DIR / "load")
    core_loads = parse_rknpu_core_loads(debug_load_raw)
    devfreq_load, devfreq_load_freq = parse_devfreq_load(_read_text(NPU_DEVFREQ_DIR / "load"))
    cur_freq_hz = _read_int(NPU_DEVFREQ_DIR / "cur_freq") or _read_int(RKNPU_DEBUG_DIR / "freq")
    min_freq_hz = _read_int(NPU_DEVFREQ_DIR / "min_freq")
    max_freq_hz = _read_int(NPU_DEVFREQ_DIR / "max_freq")
    volt_uv = _read_int(RKNPU_DEBUG_DIR / "volt")
    governor = _read_text(NPU_DEVFREQ_DIR / "governor")
    power = _read_text(RKNPU_DEBUG_DIR / "power")
    reset = _read_text(RKNPU_DEBUG_DIR / "reset")
    version = _read_text(RKNPU_DEBUG_DIR / "version")
    available = bool(core_loads or cur_freq_hz or devfreq_load is not None or version)
    max_core_load = max(core_loads.values()) if core_loads else None
    avg_core_load = (sum(core_loads.values()) / len(core_loads)) if core_loads else None
    return {
        "available": available,
        "core_loads": core_loads,
        "max_core_load": max_core_load,
        "avg_core_load": avg_core_load,
        "devfreq_load": devfreq_load,
        "devfreq_load_freq_hz": devfreq_load_freq,
        "cur_freq_hz": cur_freq_hz,
        "cur_freq_mhz": _hz_to_mhz(cur_freq_hz),
        "min_freq_hz": min_freq_hz,
        "min_freq_mhz": _hz_to_mhz(min_freq_hz),
        "max_freq_hz": max_freq_hz,
        "max_freq_mhz": _hz_to_mhz(max_freq_hz),
        "volt_uv": volt_uv,
        "volt_v": (float(volt_uv) / 1_000_000.0) if volt_uv is not None else None,
        "governor": governor,
        "power": power,
        "reset": reset,
        "version": version,
        "debug_load_raw": debug_load_raw,
    }


def npu_status_flags(status: dict) -> list:
    if not isinstance(status, dict) or not status.get("available"):
        return []
    flags = []
    core_loads = status.get("core_loads") if isinstance(status.get("core_loads"), dict) else {}
    max_core = status.get("max_core_load")
    if max_core is None and core_loads:
        try:
            max_core = max(float(value) for value in core_loads.values())
        except (TypeError, ValueError):
            max_core = None
    devfreq_load = status.get("devfreq_load")
    try:
        high_core = max_core is not None and float(max_core) >= 90.0
    except (TypeError, ValueError):
        high_core = False
    try:
        high_total = not core_loads and devfreq_load is not None and float(devfreq_load) >= 90.0
    except (TypeError, ValueError):
        high_total = False
    if high_core or high_total:
        flags.append("npu_saturated")
    return flags


def format_npu_status(status: dict) -> str:
    if not isinstance(status, dict) or not status.get("available"):
        return "unavailable"
    core_loads = status.get("core_loads") if isinstance(status.get("core_loads"), dict) else {}
    if core_loads:
        core_part = "/".join(
            f"c{idx}:{float(core_loads[idx]):.0f}%"
            for idx in sorted(core_loads.keys())
        )
    else:
        core_part = "core:-"
    freq_mhz = status.get("cur_freq_mhz")
    max_freq_mhz = status.get("max_freq_mhz")
    if freq_mhz is None:
        freq_part = "freq=-"
    elif max_freq_mhz is None:
        freq_part = f"freq={float(freq_mhz):.0f}MHz"
    else:
        freq_part = f"freq={float(freq_mhz):.0f}/{float(max_freq_mhz):.0f}MHz"
    dev_load = status.get("devfreq_load")
    dev_part = f"load={float(dev_load):.0f}%" if dev_load is not None else "load=-"
    volt_v = status.get("volt_v")
    volt_part = f"volt={float(volt_v):.3f}V" if volt_v is not None else "volt=-"
    governor = str(status.get("governor") or "-").strip() or "-"
    power = str(status.get("power") or "-").strip() or "-"
    reset = str(status.get("reset") or "-").strip() or "-"
    flags = npu_status_flags(status)
    flag_part = f" flags={','.join(flags)}" if flags else ""
    return (
        f"{core_part} {dev_part} {freq_part} {volt_part} "
        f"gov={governor} power={power} reset={reset}{flag_part}"
    )
