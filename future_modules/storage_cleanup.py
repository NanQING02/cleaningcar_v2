import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


_SNAPSHOT_SUFFIXES = ("_raw", "_annotated")
RUNTIME_STORAGE_CLEANUP_ENABLED = False


@dataclass(frozen=True)
class RetentionPolicy:
    root: Path
    keep_days: int
    keep_count: int
    label: str
    preserve_latest_groups: int = 0
    snapshot_grouping: bool = False


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _iter_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for base, _, names in os.walk(root):
        for name in names:
            files.append(Path(base) / name)
    files.sort(key=_safe_mtime)
    return files


def _snapshot_group_key(path: Path) -> str:
    stem = path.stem
    for suffix in _SNAPSHOT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _collect_protected_paths(files: Iterable[Path], policy: RetentionPolicy) -> Set[Path]:
    if policy.preserve_latest_groups <= 0:
        return set()
    if not policy.snapshot_grouping:
        ordered = sorted(files, key=_safe_mtime, reverse=True)
        return set(ordered[: policy.preserve_latest_groups])

    groups: Dict[str, Dict[str, object]] = {}
    for path in files:
        key = _snapshot_group_key(path)
        entry = groups.setdefault(key, {"mtime": 0.0, "paths": []})
        mtime = _safe_mtime(path)
        if mtime > float(entry["mtime"]):
            entry["mtime"] = mtime
        entry["paths"].append(path)

    ordered_groups = sorted(groups.values(), key=lambda item: float(item["mtime"]), reverse=True)
    protected: Set[Path] = set()
    for item in ordered_groups[: policy.preserve_latest_groups]:
        for path in item["paths"]:
            protected.add(path)
    return protected


def _remove_file(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"[storage-cleanup] warning: failed to delete {path}: {exc}")
        return False


def _cleanup_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for base, dirs, files in os.walk(root, topdown=False):
        if dirs or files:
            continue
        path = Path(base)
        if path == root:
            continue
        try:
            path.rmdir()
        except OSError:
            continue


class RuntimeStorageCleaner:
    def __init__(self, policies: Iterable[RetentionPolicy], interval_seconds: int = 600):
        deduped: List[RetentionPolicy] = []
        seen = set()
        for policy in policies:
            root = Path(policy.root).resolve()
            key = (str(root), policy.label)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                RetentionPolicy(
                    root=root,
                    keep_days=max(0, int(policy.keep_days)),
                    keep_count=max(0, int(policy.keep_count)),
                    label=policy.label,
                    preserve_latest_groups=max(0, int(policy.preserve_latest_groups)),
                    snapshot_grouping=bool(policy.snapshot_grouping),
                )
            )
        self.policies = deduped
        self.interval_seconds = max(1, int(interval_seconds))
        self._last_run_ts = 0.0

    def run_due(self, now: Optional[float] = None, force: bool = False, reason: str = "periodic") -> None:
        now_ts = float(time.time() if now is None else now)
        if not force and self._last_run_ts and now_ts - self._last_run_ts < self.interval_seconds:
            return
        self.run_once(now=now_ts, reason=reason)

    def run_once(self, now: Optional[float] = None, reason: str = "manual") -> None:
        now_ts = float(time.time() if now is None else now)
        self._last_run_ts = now_ts
        if not RUNTIME_STORAGE_CLEANUP_ENABLED:
            return
        for policy in self.policies:
            try:
                self._apply_policy(policy, now_ts, reason)
            except Exception as exc:
                print(
                    f"[storage-cleanup] warning: policy={policy.label} root={policy.root} "
                    f"reason={reason} error={exc}"
                )

    def _apply_policy(self, policy: RetentionPolicy, now_ts: float, reason: str) -> None:
        files = _iter_files(policy.root)
        if not files:
            return

        protected = _collect_protected_paths(files, policy)
        age_deleted = 0
        count_deleted = 0
        deleted_any = False
        cutoff_ts = None
        if policy.keep_days > 0:
            cutoff_ts = now_ts - (policy.keep_days * 86400)

        remaining: List[Path] = []
        for path in files:
            if path in protected:
                remaining.append(path)
                continue
            if cutoff_ts is not None and _safe_mtime(path) < cutoff_ts:
                if _remove_file(path):
                    age_deleted += 1
                    deleted_any = True
                else:
                    remaining.append(path)
                continue
            remaining.append(path)

        if policy.keep_count > 0 and len(remaining) > policy.keep_count:
            removable = [path for path in remaining if path not in protected]
            excess = len(remaining) - policy.keep_count
            for path in removable:
                if excess <= 0:
                    break
                if _remove_file(path):
                    count_deleted += 1
                    excess -= 1
                    deleted_any = True

        if deleted_any:
            _cleanup_empty_dirs(policy.root)

        if age_deleted or count_deleted:
            print(
                f"[storage-cleanup] policy={policy.label} root={policy.root} reason={reason} "
                f"scanned={len(files)} deleted_age={age_deleted} deleted_count={count_deleted} "
                f"protected={len(protected)}"
            )
