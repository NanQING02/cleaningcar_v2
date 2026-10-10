"""aarch64/Linux原生契约测试，使用假的librga，不提交硬件任务。"""
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(platform.system() == 'Linux' and Path('/usr/include/rga/drmrga.h').exists(), '需要板端RGA头文件和gcc')
class NativeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        cls.guard = cls.work / 'guard.so'
        subprocess.run(['gcc', '-shared', '-fPIC', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-o', str(cls.guard), str(ROOT / 'native/rga3_guard/guard.c'), '-ldl'], check=True)
        cls.lib = cls.work / 'libfake.so'
        lib_src = cls.work / 'lib.c'
        lib_src.write_text('''
#include <errno.h>
#include <rga/RgaApi.h>
static int calls=0, s_core=0, d_core=0;
int get_calls(void){return calls;}
int get_source_core(void){return s_core;}
int get_dest_core(void){return d_core;}
int c_RkRgaBlit(rga_info_t *s,rga_info_t *d,rga_info_t *p){
    (void)p; calls++; s_core=s->core; d_core=d->core;
    d->out_fence_fd=88; errno=EIO; return -EIO;
}
''')
        subprocess.run(['gcc', '-shared', '-fPIC', '-o', str(cls.lib), str(lib_src)], check=True)
        cls.plugin = cls.work / 'plugin.so'
        plugin_src = cls.work / 'plugin.c'
        plugin_src.write_text('''
#include <string.h>
#include <rga/RgaApi.h>
#include <rga/rga.h>
static int restored=0, fence=0;
int get_restored(void){return restored;}
int get_fence(void){return fence;}
int convert(int mode){
    rga_info_t s={0},d={0};
    s.fd=10;d.fd=11;s.core=2;d.core=4;
    rga_set_rect(&s.rect,0,0,1920,1080,1920,1088,RK_FORMAT_YCbCr_420_SP);
    rga_set_rect(&d.rect,0,0,1920,1080,1920,1080,RK_FORMAT_BGR_888);
    if(mode==1)s.virAddr=(void*)1;
    if(mode==2)d.rect.format=RK_FORMAT_RGBA_8888;
    if(mode==3)d.rect.wstride=1919;
    int rc=c_RkRgaBlit(&s,&d,0);
    restored=(s.core==2&&d.core==4);fence=d.out_fence_fd;
    return rc;
}
''')
        subprocess.run(['gcc', '-shared', '-fPIC', '-o', str(cls.plugin), str(plugin_src),
                        '-L' + str(cls.work), '-lfake', '-Wl,-rpath,' + str(cls.work)], check=True)
        cls.env = dict(os.environ, LD_PRELOAD=str(cls.guard), CLEANINGCAR_RGA3_ACTIVE='1',
                       CLEANINGCAR_RGA3_NONCE='test', CLEANINGCAR_RGA3_LIB=str(cls.lib),
                       CLEANINGCAR_RGA3_PLUGIN=str(cls.plugin))

    def run_code(self, code, env=None):
        import sys
        return subprocess.run([sys.executable, '-c', code], env=self.env if env is None else env,
                              text=True, capture_output=True)

    def test_known_job_core_restoration_and_error_fence_preserved(self):
        result = self.run_code(f'''
import ctypes,errno
p=ctypes.CDLL({str(self.plugin)!r},use_errno=True);l=ctypes.CDLL({str(self.lib)!r})
assert p.convert(0)==-errno.EIO
assert ctypes.get_errno()==errno.EIO
assert l.get_calls()==1
assert l.get_source_core()==3 and l.get_dest_core()==3
assert p.get_restored()==1 and p.get_fence()==88
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('guarded=1', result.stderr)
        self.assertIn('failed=1', result.stderr)

    def test_unknown_jobs_refused_without_hardware_fallback(self):
        result = self.run_code(f'''
import ctypes,errno
p=ctypes.CDLL({str(self.plugin)!r});l=ctypes.CDLL({str(self.lib)!r})
for mode in (1,2,3): assert p.convert(mode)==-errno.EINVAL
assert l.get_calls()==0
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('rejected=3', result.stderr)

    def test_unrelated_caller_preserves_original_core(self):
        caller_src = self.work / 'foreign.c'
        caller_src.write_text('#include <rga/RgaApi.h>\nint foreign(void){rga_info_t s={0},d={0};s.core=7;d.core=4;return c_RkRgaBlit(&s,&d,0);}\n')
        caller = self.work / 'foreign.so'
        subprocess.run(['gcc', '-shared', '-fPIC', '-o', str(caller), str(caller_src),
                        '-L' + str(self.work), '-lfake', '-Wl,-rpath,' + str(self.work)], check=True)
        result = self.run_code(f'''
import ctypes,errno
p=ctypes.CDLL({str(caller)!r});l=ctypes.CDLL({str(self.lib)!r})
assert p.foreign()==-errno.EIO
assert l.get_source_core()==7 and l.get_dest_core()==4
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('foreign=1', result.stderr)
        self.assertIn('guarded=0', result.stderr)

    def test_child_exec_does_not_inherit_preload(self):
        result = self.run_code('''
import os,subprocess,sys
assert "LD_PRELOAD" not in os.environ
assert os.environ['CLEANINGCAR_RGA3_READY']=='test'
subprocess.run([sys.executable,'-c','import ctypes; assert not hasattr(ctypes.CDLL(None), "cleaningcar_rga3_guard_api_version")'],check=True)
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_incomplete_native_context_exits_before_python(self):
        env = dict(self.env)
        del env['CLEANINGCAR_RGA3_NONCE']
        result = self.run_code('print("unsafe")', env)
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertNotIn('unsafe', result.stdout)


@unittest.skipUnless(platform.system() == 'Linux' and (ROOT / '.local/rga3_guard/manifest.json').exists(), '需要板端已构建产物')
class BoardVersionGuardTests(unittest.TestCase):
    def setUp(self):
        import shutil
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / '.local/rga3_guard', self.root / '.local/rga3_guard')
        import json
        self.manifest = json.loads((self.root / '.local/rga3_guard/manifest.json').read_text())
        (self.root / 'native/rga3_guard').mkdir(parents=True)
        shutil.copy2(ROOT / 'native/rga3_guard/guard.c', self.root / 'native/rga3_guard/guard.c')

    def change_manifest(self, key, value):
        import json
        path = self.root / '.local/rga3_guard/manifest.json'
        data = dict(self.manifest)
        data[key] = value
        path.write_text(json.dumps(data))

    def test_validated_system_stack_passes(self):
        from cleaningcar.rga3_guard import validate
        validate(self.root)

    def test_changed_source_is_refused(self):
        from cleaningcar.rga3_guard import GuardError, validate
        with (self.root / 'native/rga3_guard/guard.c').open('a') as f:
            f.write('\n/* changed */\n')
        with self.assertRaises(GuardError):
            validate(self.root)

    def test_changed_binary_is_refused(self):
        from cleaningcar.rga3_guard import GuardError, validate
        with (self.root / '.local/rga3_guard/librga3_guard.so').open('ab') as f:
            f.write(b'changed')
        with self.assertRaises(GuardError):
            validate(self.root)

    def test_unverified_library_or_abi_is_refused(self):
        from cleaningcar.rga3_guard import GuardError, validate
        for key, value in (('librga', {'path': '/usr/lib/unknown', 'sha256': 'bad'}),
                           ('abi', {'size': 696, 'core_offset': 0}),
                           ('driver_path', '/tmp/fake_driver'), ('header_sha256', {})):
            with self.subTest(key=key):
                self.change_manifest(key, value)
                with self.assertRaises(GuardError):
                    validate(self.root)

    def test_bootstrap_reexec_keeps_pid_group_and_launch_id(self):
        import json
        import sys
        config = self.root / 'configs/config.json'
        config.parent.mkdir()
        config.write_text('{}')
        (self.root / '.local/rga3_guard/enabled.json').write_text(json.dumps({'schema': 1, 'configs': ['configs/config.json']}))
        script = self.root / 'smoke.py'
        script.write_text(f'''
import os,sys,json
sys.path.insert(0,{str(ROOT)!r})
from cleaningcar.rga3_guard import bootstrap
print(json.dumps({{'phase':'before','pid':os.getpid(),'pgid':os.getpgid(0),'launch':os.environ['CLEANINGCAR_LAUNCH_ID']}}),flush=True)
bootstrap({str(self.root)!r})
print(json.dumps({{'phase':'after','pid':os.getpid(),'pgid':os.getpgid(0),'launch':os.environ['CLEANINGCAR_LAUNCH_ID'],'preload':os.environ.get('LD_PRELOAD')}}),flush=True)
''')
        result = subprocess.run([sys.executable, '-u', str(script), '--config', str(config)],
                                env=dict(os.environ, CLEANINGCAR_LAUNCH_ID='unchanged'),
                                text=True, capture_output=True, start_new_session=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        self.assertEqual([r['phase'] for r in rows], ['before', 'before', 'after'])
        self.assertEqual(len({r['pid'] for r in rows}), 1)
        self.assertEqual(len({r['pgid'] for r in rows}), 1)
        self.assertTrue(all(r['launch']=='unchanged' for r in rows))
        self.assertIsNone(rows[-1]['preload'])


if __name__ == '__main__':
    unittest.main()
