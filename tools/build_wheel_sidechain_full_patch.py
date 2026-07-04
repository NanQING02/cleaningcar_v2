#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "packages"
PATCH_NAME = "cleaningcar_wheel_sidechain_full_patch"

FILES = [
    "config_manager.py",
    "cleaningcar/events.py",
    "cleaningcar/wheel.py",
    "tests/test_config_global_video_cleanup.py",
    "tests/test_wheel_photo.py",
    "tools/build_wheel_sidechain_full_patch.py",
    "tools/mock_wheel_photo_api.py",
    "tools/run_wheel_probe_board.sh",
    "tools/wheel_sidechain_probe.py",
    "tools/wheel_sidechain_test_README.md",
    "docs/部署验收/车轮旁支独立测试说明.md",
]

ALIASES = [
    (
        "docs/部署验收/车轮旁支独立测试说明.md",
        "docs/wheel_sidechain_independent_test.md",
    ),
]

README = """# CleaningCar Wheel Sidechain Full Patch

This archive contains the modified wheel-sidechain source files, regression tests,
board-side probe, PC-side mock API, and acceptance documentation.

## Apply on board

Extract the archive outside or inside the CleaningCar project, then copy its
contents into the existing project root:

```bash
tar -xzf cleaningcar_wheel_sidechain_full_patch.tar.gz
cp -a cleaningcar_wheel_sidechain_full_patch/. /path/to/cleaningcar_v2/
```

Optional backup before overwriting:

```bash
cd /path/to/cleaningcar_v2
mkdir -p ../cleaningcar_wheel_backup
cp -a config_manager.py cleaningcar/events.py cleaningcar/wheel.py tools ../cleaningcar_wheel_backup/
```

## Board probe

```bash
cd /path/to/cleaningcar_v2
chmod +x tools/run_wheel_probe_board.sh
./tools/run_wheel_probe_board.sh \\
  --config configs/config.json \\
  --wheel-photo-url http://<PC_IP>:28014/api/vehicle/wheel-photo \\
  --duration 120 \\
  --target-fps 10 \\
  --force-event-driven false
```

## PC mock API

```bash
python tools/mock_wheel_photo_api.py --host 0.0.0.0 --port 28014
```
"""


def _copy_files(dst_root: Path) -> None:
    package_root = dst_root / PATCH_NAME
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst = package_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for src_rel, dst_rel in ALIASES:
        src = ROOT / src_rel
        dst = package_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (package_root / "PATCH_README.md").write_text(README, encoding="utf-8", newline="\n")


def _tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    stat = path.stat()
    info.size = stat.st_size
    info.mtime = int(stat.st_mtime)
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
    elif path.suffix in {".py", ".sh"}:
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def build_tar_gz(staging_root: Path) -> Path:
    out_path = PACKAGE_DIR / f"{PATCH_NAME}.tar.gz"
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(out_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted((staging_root / PATCH_NAME).rglob("*")):
            arcname = path.relative_to(staging_root).as_posix()
            info = _tar_info(path, arcname)
            if path.is_file():
                with path.open("rb") as f:
                    tf.addfile(info, f)
            else:
                tf.addfile(info)
    return out_path


def build_zip(staging_root: Path) -> Path:
    out_path = PACKAGE_DIR / f"{PATCH_NAME}.zip"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((staging_root / PATCH_NAME).rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(staging_root).as_posix()
            zf.write(path, arcname)
    return out_path


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging_root = Path(tmp)
        _copy_files(staging_root)
        tar_path = build_tar_gz(staging_root)
        zip_path = build_zip(staging_root)
    print(tar_path)
    print(zip_path)


if __name__ == "__main__":
    main()
