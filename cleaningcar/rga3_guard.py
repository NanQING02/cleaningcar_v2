"""推理启动前的RGA3灰度预检查；只依赖标准库，不导入解码/NPU模块。"""
import hashlib
import json
import os
import platform
import struct
import sys
from pathlib import Path
from uuid import uuid4


STATE_REL = Path('.local/rga3_guard')
LIB_SHA256 = '85deb0d51204c1b78ee069c9c1737f99c481f223d831d8538e289fbc7b000668'
PLUGIN_SHA256 = '7c821fcc3cd4ea6be15c5220b12a0271b3b7b9cd51541e265247fb23ce4b55b9'
DRIVER_VERSION = 'v1.3.10i'
HEADER_SHA256 = {
    '/usr/include/rga/drmrga.h': 'c2a6f0bca03619aa58c558635b58b3cc4ed3b7e82c3f23bfc474f98c2b6968d1',
    '/usr/include/rga/rga.h': 'a79d9586d1c96d5da70a062de5fcb7a0bf9e95821fc44f213f92543b0b8d15eb',
}


class GuardError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise GuardError(f'无法读取RGA3灰度文件：{path}: {exc}') from exc
    if not isinstance(data, dict) or data.get('schema') != 1:
        raise GuardError(f'RGA3灰度文件schema错误：{path}')
    return data


def validate(root):
    root = Path(root).resolve()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise GuardError('RGA3灰度仅支持已核验的Linux aarch64设备')
    compatible = Path('/proc/device-tree/compatible').read_bytes()
    if not set(compatible.split(b'\x00')) & {b'rockchip,rk3588', b'rockchip,rk3588s'}:
        raise GuardError('设备树不匹配RK3588')
    manifest = read_json(root / STATE_REL / 'manifest.json')
    if manifest.get('abi') != {'size': 696, 'core_offset': 188}:
        raise GuardError('RGA3构建ABI不匹配')
    if manifest.get('header_sha256') != HEADER_SHA256:
        raise GuardError('RGA3构建头文件不在已验证列表内')
    for key, expected in (('librga', LIB_SHA256), ('plugin', PLUGIN_SHA256)):
        item = manifest.get(key, {})
        path = Path(item.get('path', '')).resolve()
        if not str(path).startswith('/usr/lib/'):
            raise GuardError(f'{key}必须是厂家系统库路径')
        if item.get('sha256') != expected or sha256(path) != expected:
            raise GuardError(f'{key}版本已变化，拒绝启用RGA3灰度')
    driver_path = Path(manifest.get('driver_path', ''))
    if driver_path not in (Path('/sys/kernel/debug/rkrga/driver_version'), Path('/proc/rkrga/driver_version')):
        raise GuardError('RGA驱动版本路径不合法')
    if driver_path.read_text().strip() != f'RGA multicore Device Driver: {DRIVER_VERSION}':
        raise GuardError('RGA驱动版本变化，拒绝启用')
    library = (root / STATE_REL / 'librga3_guard.so').resolve()
    if library.parent != (root / STATE_REL).resolve() or sha256(library) != manifest.get('guard_sha256'):
        raise GuardError('项目RGA3预加载库缺失或哈希不匹配')
    with library.open('rb') as f:
        elf = f.read(20)
    if len(elf) != 20 or elf[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', elf, 18)[0] != 183:
        raise GuardError('预加载库不是aarch64 ELF64')
    for key in ('source_sha256', 'header_sha256'):
        if key not in manifest:
            raise GuardError(f'构建证据缺少{key}')
    if sha256(root / 'native/rga3_guard/guard.c') != manifest['source_sha256']:
        raise GuardError('RGA3源码已变化，请先重新构建')
    return manifest, library


def prepare_environment(root, env=None):
    manifest, library = validate(root)
    result = dict(os.environ if env is None else env)
    if result.get('LD_PRELOAD', '').strip():
        raise GuardError('已有LD_PRELOAD，拒绝叠加未知拦截库')
    result['LD_PRELOAD'] = str(library)
    result['CLEANINGCAR_RGA3_ACTIVE'] = '1'
    result['CLEANINGCAR_RGA3_NONCE'] = uuid4().hex
    result['CLEANINGCAR_RGA3_LIB'] = manifest['librga']['path']
    result['CLEANINGCAR_RGA3_PLUGIN'] = manifest['plugin']['path']
    return result


def selected(root, argv):
    root = Path(root).resolve()
    switch = root / STATE_REL / 'enabled.json'
    if not switch.exists():
        return False
    data = read_json(switch)
    targets = data.get('configs', [])
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise GuardError('RGA3灰度配置列表无效')
    config = root / 'configs/config.json'
    for i, arg in enumerate(argv):
        if arg == '--config' and i + 1 < len(argv):
            config = Path(argv[i + 1])
        elif arg.startswith('--config='):
            config = Path(arg.split('=', 1)[1])
    return config.resolve() in [(root / t).resolve() for t in targets]


def bootstrap(root):
    """旧Web也能触发：脚本在导入cli前exec一次，PID/launch_id保持不变。"""
    nonce = os.environ.get('CLEANINGCAR_RGA3_NONCE')
    if nonce:
        if os.environ.get('CLEANINGCAR_RGA3_READY') != nonce:
            raise GuardError('预加载库未完成初始化，拒绝启动推理')
        import ctypes
        try:
            api = ctypes.CDLL(None).cleaningcar_rga3_guard_api_version
            api.restype = ctypes.c_int
            if api() != 1:
                raise GuardError('RGA3预加载库API不匹配')
        except AttributeError as exc:
            raise GuardError('RGA3库未映射进当前PID') from exc
        # 子进程不会继承LD_PRELOAD；保留READY用于本PID审计。
        for key in ('CLEANINGCAR_RGA3_NONCE', 'CLEANINGCAR_RGA3_ACTIVE', 'CLEANINGCAR_RGA3_LIB', 'CLEANINGCAR_RGA3_PLUGIN'):
            os.environ.pop(key, None)
        print('[rga3-guard] 当前推理PID已启用，子进程不继承预加载', flush=True)
        return
    if not selected(root, sys.argv[1:]):
        return
    env = prepare_environment(root)
    args = [sys.executable, *getattr(sys, 'orig_argv', [sys.executable, *sys.argv])[1:]]
    os.execve(sys.executable, args, env)
