#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "packages"
DEFAULT_NAME = "cleaningcar_v2_runtime"

EXCLUDE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}
EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
}
EXCLUDE_PARTS = {
    "events",
    "video_result",
    "ftp",
    "mock_api_output",
    "wheel_probe_output",
}
EXCLUDE_PACKAGE_PREFIXES = (
    "packages/cleaningcar_wheel_sidechain_full_patch.",
    "packages/cleaningcar_v2_runtime.",
)


def _git_files() -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    output = result.stdout.decode("utf-8")
    return [item for item in output.split("\0") if item]


def _include_file(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    path = Path(rel)
    if any(part in EXCLUDE_NAMES for part in path.parts):
        return False
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if rel.startswith(EXCLUDE_PACKAGE_PREFIXES):
        return False
    return True


def _tar_info(src: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    stat = src.stat()
    info.size = stat.st_size
    info.mtime = int(stat.st_mtime)
    if src.suffix in {".sh", ".py"}:
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def build_tar_gz(package_name: str, files: list[str]) -> Path:
    out_path = PACKAGE_DIR / f"{package_name}.tar.gz"
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(out_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for rel in files:
            src = ROOT / rel
            arcname = f"{package_name}/{rel.replace('\\', '/')}"
            info = _tar_info(src, arcname)
            with src.open("rb") as f:
                tf.addfile(info, f)
    return out_path


def build_zip(package_name: str, files: list[str]) -> Path:
    out_path = PACKAGE_DIR / f"{package_name}.zip"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            src = ROOT / rel
            arcname = f"{package_name}/{rel.replace('\\', '/')}"
            zf.write(src, arcname)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full CleaningCar runtime package from tracked files.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Package root/output base name.")
    parser.add_argument("--zip", action="store_true", help="Also build a zip package.")
    args = parser.parse_args()

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    files = [rel for rel in _git_files() if _include_file(rel)]
    if not files:
        raise SystemExit("no files to package")
    tar_path = build_tar_gz(args.name, files)
    print(tar_path)
    if args.zip:
        print(build_zip(args.name, files))


if __name__ == "__main__":
    main()
