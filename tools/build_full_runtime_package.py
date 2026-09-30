#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "packages"
DEFAULT_NAME = "cleaningcar_v2_runtime"
EXTRA_FILES = ("requirements.lock",)

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


def _runtime_files() -> list[str]:
    files = set(_git_files())
    for rel in EXTRA_FILES:
        if (ROOT / rel).is_file():
            files.add(rel)
    wheelhouse = PACKAGE_DIR / "wheelhouse"
    if wheelhouse.is_dir():
        for path in wheelhouse.rglob("*"):
            if path.is_file():
                files.add(path.relative_to(ROOT).as_posix())
    return sorted(files)


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


def _read_normalized(rel: str) -> bytes:
    """读取文件内容；.sh 统一转为 LF 行尾。

    Windows 工作区可能因 core.autocrlf 被检出为 CRLF，直接打包会让
    脚本在 Linux 上报 "/usr/bin/env: bash\\r" 而无法执行；在打包源头
    归一化，保证无论在哪个平台构建交付包都是 LF。
    """
    src = ROOT / rel
    data = src.read_bytes()
    if rel.endswith(".sh"):
        data = data.replace(b"\r\n", b"\n")
    return data


def build_tar_gz(package_name: str, files: list[str]) -> Path:
    out_path = PACKAGE_DIR / f"{package_name}.tar.gz"
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(out_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for rel in files:
            src = ROOT / rel
            archive_rel = rel.replace("\\", "/")
            arcname = f"{package_name}/{archive_rel}"
            info = _tar_info(src, arcname)
            data = _read_normalized(rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return out_path


def build_zip(package_name: str, files: list[str]) -> Path:
    out_path = PACKAGE_DIR / f"{package_name}.zip"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            archive_rel = rel.replace("\\", "/")
            arcname = f"{package_name}/{archive_rel}"
            data = _read_normalized(rel)
            info = zipfile.ZipInfo(arcname)
            mode = 0o755 if rel.endswith((".sh", ".py")) else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return out_path


def validate_tar_gz(path: Path, package_name: str) -> None:
    required = {
        f"{package_name}/configs/config.json",
        f"{package_name}/configs/config_绕行.json",
        f"{package_name}/requirements.lock",
    }
    with tarfile.open(path, "r:gz") as tf:
        names = set(tf.getnames())
        # .sh必须是LF行尾：CRLF会让脚本在Linux上直接无法执行
        for member in tf.getmembers():
            if member.name.endswith(".sh"):
                data = tf.extractfile(member).read()
                if b"\r" in data:
                    raise RuntimeError(f"CRLF found in packaged script: {member.name}")
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"archive missing required UTF-8 paths: {missing}")
    for name in names:
        name.encode("utf-8", errors="strict")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full CleaningCar runtime package from tracked files.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Package root/output base name.")
    parser.add_argument("--zip", action="store_true", help="Also build a zip package.")
    args = parser.parse_args()

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    files = [rel for rel in _runtime_files() if _include_file(rel)]
    if not files:
        raise SystemExit("no files to package")
    tar_path = build_tar_gz(args.name, files)
    validate_tar_gz(tar_path, args.name)
    print(f"{tar_path} sha256={sha256_file(tar_path)}")
    if args.zip:
        print(build_zip(args.name, files))


if __name__ == "__main__":
    main()
