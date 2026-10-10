import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cleaningcar.rga3_guard import GuardError, STATE_REL, bootstrap, prepare_environment, selected


class Rga3GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / STATE_REL
        self.state.mkdir(parents=True)

    def switch(self, data):
        (self.state / 'enabled.json').write_text(json.dumps(data), encoding='utf-8')

    def test_default_disabled_without_reading_native_files(self):
        self.assertFalse(selected(self.root, ['--config', str(self.root / 'configs/config.json')]))
        with patch.dict(os.environ, {}, clear=True), patch('cleaningcar.rga3_guard.validate') as validate:
            bootstrap(self.root)
            validate.assert_not_called()

    def test_selected_config_only_and_equal_form(self):
        self.switch({'schema': 1, 'configs': ['configs/config.json']})
        self.assertTrue(selected(self.root, ['--config=' + str(self.root / 'configs/config.json')]))
        self.assertFalse(selected(self.root, ['--config', str(self.root / 'configs/config_绕行.json')]))

    def test_malformed_switch_refused(self):
        self.switch({'schema': 1, 'configs': 'all'})
        with self.assertRaises(GuardError):
            selected(self.root, [])

    @patch('cleaningcar.rga3_guard.validate')
    def test_injected_environment_does_not_mutate_web(self, validate):
        validate.return_value = ({'librga': {'path': '/usr/lib/librga.so.2'}, 'plugin': {'path': '/usr/lib/plugin.so'}}, Path('/project/guard.so'))
        original = {'CLEANINGCAR_LAUNCH_ID': 'keep', 'PATH': '/bin'}
        env = prepare_environment(self.root, original)
        self.assertEqual(original, {'CLEANINGCAR_LAUNCH_ID': 'keep', 'PATH': '/bin'})
        self.assertEqual(env['CLEANINGCAR_LAUNCH_ID'], 'keep')
        self.assertEqual(env['LD_PRELOAD'], str(Path('/project/guard.so')))
        with self.assertRaises(GuardError):
            prepare_environment(self.root, {'LD_PRELOAD': '/unknown.so'})

    def test_missing_library_cannot_silently_start(self):
        with patch.dict(os.environ, {'CLEANINGCAR_RGA3_NONCE': 'n'}, clear=True):
            with self.assertRaises(GuardError):
                bootstrap(self.root)

    @patch('cleaningcar.rga3_guard.platform.system', return_value='Windows')
    def test_incompatible_machine_refused(self, system):
        with self.assertRaises(GuardError):
            prepare_environment(self.root, {})

    @patch('cleaningcar.rga3_guard.os.execve')
    @patch('cleaningcar.rga3_guard.prepare_environment', return_value={'TEST': '1'})
    @patch('cleaningcar.rga3_guard.selected', return_value=True)
    def test_exec_uses_original_interpreter_arguments(self, selected_mock, prepare, execve):
        import sys
        args = [sys.executable, '-u', 'run_zone_detect.py', '--config', '/tmp/config.json']
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'orig_argv', args):
            bootstrap(self.root)
        execve.assert_called_once_with(sys.executable, args, {'TEST': '1'})


if __name__ == '__main__':
    unittest.main()
