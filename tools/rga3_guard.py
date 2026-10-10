#!/usr/bin/env python3
"""构建、检查和控制本地RGA3灰度库；不安装到系统目录。"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cleaningcar.rga3_guard import (  # noqa: E402
    DRIVER_VERSION, HEADER_SHA256, LIB_SHA256, PLUGIN_SHA256, STATE_REL, GuardError,
    prepare_environment, sha256, validate,
)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def build():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise GuardError('请在目标RK3588设备上原生构建')
    lib = Path('/lib/aarch64-linux-gnu/librga.so.2').resolve()
    plugin = Path('/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstrockchipmpp.so').resolve()
    if sha256(lib) != LIB_SHA256 or sha256(plugin) != PLUGIN_SHA256:
        raise GuardError('厂家librga/gst版本不在已验证列表中')
    if any(sha256(path) != expected for path, expected in HEADER_SHA256.items()):
        raise GuardError('厂家RGA头文件版本不匹配，拒绝构建')
    driver = next((p for p in (Path('/sys/kernel/debug/rkrga/driver_version'), Path('/proc/rkrga/driver_version')) if p.exists()), None)
    if driver is None or driver.read_text().strip() != f'RGA multicore Device Driver: {DRIVER_VERSION}':
        raise GuardError('RGA驱动版本不匹配')
    state = ROOT / STATE_REL
    if (state / 'enabled.json').exists():
        raise GuardError('请先关闭灰度，不能在线覆盖已启用的库')
    state.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'native/rga3_guard/guard.c'
    target = state / 'librga3_guard.so'
    temporary = target.with_suffix('.tmp.so')
    subprocess.run(['gcc', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-std=c11', '-Wl,-z,relro,-z,now', '-o', str(temporary), str(source), '-ldl'], check=True)
    os.replace(temporary, target)
    headers = [Path('/usr/include/rga/drmrga.h'), Path('/usr/include/rga/rga.h')]
    manifest = {
        'schema': 1, 'abi': {'size': 696, 'core_offset': 188},
        'librga': {'path': str(lib), 'sha256': sha256(lib)},
        'plugin': {'path': str(plugin), 'sha256': sha256(plugin)},
        'driver_path': str(driver), 'source_sha256': sha256(source),
        'header_sha256': {str(h): sha256(h) for h in headers},
        'guard_sha256': sha256(target),
    }
    write_json(state / 'manifest.json', manifest)
    check()
    print('构建成功；灰度默认关闭')


def check():
    env = prepare_environment(ROOT)
    # 使用自己的API检查动态加载，完全不调用RGA。
    code = ('import ctypes,os; '
            'assert os.environ.get("CLEANINGCAR_RGA3_READY")==os.environ["CLEANINGCAR_RGA3_NONCE"]; '
            'assert not os.environ.get("LD_PRELOAD"); '
            'assert ctypes.CDLL(None).cleaningcar_rga3_guard_api_version()==1; '
            'print("RGA3预检查通过，未提交任何RGA任务")')
    subprocess.run([sys.executable, '-c', code], env=env, check=True)


def enable(configs):
    check()
    names = []
    for rel in configs:
        path = (ROOT / rel).resolve()
        if path.parent != (ROOT / 'configs').resolve() or path.suffix != '.json' or not path.is_file():
            raise GuardError('只能指定本项目configs内存在的配置')
        json.loads(path.read_text(encoding='utf-8'))
        names.append(path.relative_to(ROOT).as_posix())
    write_json(ROOT / STATE_REL / 'enabled.json', {'schema': 1, 'configs': sorted(set(names))})
    print('灰度名单已保存；仅下次启动这些配置的推理时生效：' + ', '.join(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('build')
    sub.add_parser('check')
    sub.add_parser('status')
    sub.add_parser('disable')
    sub.add_parser('enable').add_argument('--configs', nargs='+', required=True)
    sub.add_parser('run').add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == 'build':
            build()
        elif args.action == 'check':
            check()
        elif args.action == 'enable':
            enable(args.configs)
        elif args.action == 'disable':
            (ROOT / STATE_REL / 'enabled.json').unlink(missing_ok=True)
            print('灰度已关闭；正在运行的PID需正常重启推理才会卸载')
        elif args.action == 'status':
            switch = ROOT / STATE_REL / 'enabled.json'
            print(switch.read_text(encoding='utf-8') if switch.exists() else '灰度关闭')
        else:
            cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
            if not cmd:
                raise GuardError('run需要指定独立测试命令')
            env = prepare_environment(ROOT)
            # 子测试进程继承lib，但它再启动ffmpeg等时不会继承。
            raise SystemExit(subprocess.run(cmd, env=env).returncode)
    except (GuardError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'[rga3-guard] 操作失败：{exc}') from exc


if __name__ == '__main__':
    main()
