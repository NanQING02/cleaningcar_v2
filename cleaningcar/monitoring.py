import os
import time
from pathlib import Path

from .npu_monitor import format_npu_status, npu_status_flags, snapshot_npu_status

try:
    import psutil
except ImportError:
    psutil = None


def _safe_percent(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_mb(value):
    try:
        return float(value) / (1024 * 1024)
    except Exception:
        return 0.0


def _read_psi_avg10(name):
    path = Path("/proc/pressure") / name
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("some "):
                continue
            for item in line.split():
                if item.startswith("avg10="):
                    return float(item.split("=", 1)[1])
    except Exception:
        return None
    return None


def _read_throttle_flags():
    paths = (
        Path("/sys/devices/platform/ff9a0000.gpu/devfreq/ff9a0000.gpu/throttle"),
        Path("/sys/devices/system/cpu/cpufreq/policy0/throttle"),
    )
    flags = []
    for path in paths:
        try:
            if path.exists():
                value = path.read_text(encoding="utf-8", errors="ignore").strip()
                if value and value not in ("0", "0\n"):
                    flags.append(f"{path.name}={value}")
        except Exception:
            continue
    return ",".join(flags)


def _snapshot_children(proc):
    result = {
        "count": 0,
        "ffmpeg_count": 0,
        "cpu": 0.0,
        "rss_mb": 0.0,
    }
    if proc is None:
        return result
    try:
        children = proc.children(recursive=True)
    except Exception:
        return result
    result["count"] = len(children)
    for child in children:
        try:
            name = (child.name() or "").lower()
            cmdline = " ".join(child.cmdline()).lower()
            if "ffmpeg" in name or "ffmpeg" in cmdline:
                result["ffmpeg_count"] += 1
            result["cpu"] += _safe_percent(child.cpu_percent(interval=None))
            result["rss_mb"] += _safe_mb(child.memory_info().rss)
        except Exception:
            continue
    return result


def _snapshot_temp():
    max_temp = None
    max_label = ""
    try:
        temps = psutil.sensors_temperatures() if psutil is not None else {}
    except Exception:
        temps = {}
    for group, entries in (temps or {}).items():
        for entry in entries or []:
            try:
                current = float(entry.current)
            except Exception:
                continue
            if max_temp is None or current > max_temp:
                max_temp = current
                label = getattr(entry, "label", "") or group
                max_label = str(label)
    return max_temp, max_label


def _resource_risks(cpu, max_core, load_per_cpu, mem, swap, temp, psi_cpu, psi_mem, psi_io):
    risks = []
    if cpu >= 90.0 or max_core >= 98.0 or load_per_cpu >= 1.5 or (psi_cpu is not None and psi_cpu >= 50.0):
        risks.append("cpu_pressure")
    if mem >= 90.0 or swap >= 10.0 or (psi_mem is not None and psi_mem >= 10.0):
        risks.append("memory_pressure")
    if psi_io is not None and psi_io >= 10.0:
        risks.append("io_pressure")
    if temp is not None and temp >= 80.0:
        risks.append("thermal_high")
    return ",".join(risks) if risks else "ok"


def monitor_loop(interval, stop_event):
    if interval <= 0:
        return
    if psutil is None:
        print('[monitor] psutil not installed, monitoring disabled.')
        return
    proc = None
    try:
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)
    except Exception:
        proc = None
    started = time.time()
    count = 0
    sum_cpu = 0.0
    sum_mem = 0.0
    max_cpu = 0.0
    max_mem = 0.0
    while not stop_event.is_set():
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        cpu = sum(per_cpu) / max(1, len(per_cpu)) if per_cpu else psutil.cpu_percent(interval=None)
        max_core = max(per_cpu) if per_cpu else cpu
        vm = psutil.virtual_memory()
        mem = vm.percent
        swap = psutil.swap_memory().percent
        rss_mb = 0.0
        proc_cpu = 0.0
        proc_threads = 0
        proc_fds = 0
        if proc is not None:
            try:
                rss_mb = _safe_mb(proc.memory_info().rss)
                proc_cpu = _safe_percent(proc.cpu_percent(interval=None))
                proc_threads = int(proc.num_threads())
                if hasattr(proc, "num_fds"):
                    proc_fds = int(proc.num_fds())
            except Exception:
                rss_mb = 0.0
        child_stats = _snapshot_children(proc)
        count += 1
        sum_cpu += cpu
        sum_mem += mem
        if cpu > max_cpu:
            max_cpu = cpu
        if mem > max_mem:
            max_mem = mem
        avg_cpu = sum_cpu / max(1, count)
        avg_mem = sum_mem / max(1, count)
        uptime = time.time() - started
        cpu_count = psutil.cpu_count() or 1
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            load1 = load5 = load15 = 0.0
        load_per_cpu = load1 / max(1, cpu_count)
        psi_cpu = _read_psi_avg10("cpu")
        psi_mem = _read_psi_avg10("memory")
        psi_io = _read_psi_avg10("io")
        temp, temp_label = _snapshot_temp()
        npu_status = snapshot_npu_status()
        npu_flags = npu_status_flags(npu_status)
        risk = _resource_risks(cpu, max_core, load_per_cpu, mem, swap, temp, psi_cpu, psi_mem, psi_io)
        if npu_flags:
            risk = ",".join(([risk] if risk != "ok" else []) + npu_flags)
        msg = (
            f'[monitor] risk={risk} cpu={cpu:.1f}% max_core={max_core:.1f}% '
            f'load1={load1:.2f} load_per_cpu={load_per_cpu:.2f} '
            f'mem={mem:.1f}% avail={_safe_mb(vm.available):.0f}MB swap={swap:.1f}% '
            f'proc_cpu={proc_cpu:.1f}% rss={rss_mb:.1f}MB threads={proc_threads} fds={proc_fds} '
            f'children={child_stats["count"]} ffmpeg={child_stats["ffmpeg_count"]} '
            f'child_cpu={child_stats["cpu"]:.1f}% child_rss={child_stats["rss_mb"]:.1f}MB '
            f'avg_cpu={avg_cpu:.1f}% max_cpu={max_cpu:.1f}% '
            f'avg_mem={avg_mem:.1f}% max_mem={max_mem:.1f}% uptime={uptime:.0f}s '
            f'npu=[{format_npu_status(npu_status)}]'
        )
        if temp is not None:
            msg += f' temp={temp:.1f}C temp_label={temp_label}'
        if psi_cpu is not None or psi_mem is not None or psi_io is not None:
            msg += (
                f' psi_cpu10={psi_cpu if psi_cpu is not None else -1:.2f}'
                f' psi_mem10={psi_mem if psi_mem is not None else -1:.2f}'
                f' psi_io10={psi_io if psi_io is not None else -1:.2f}'
            )
        throttle = _read_throttle_flags()
        if throttle:
            msg += f' throttle={throttle}'
        print(msg, flush=True)
        stop_event.wait(interval)
