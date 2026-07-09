import signal
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from web.inference import InferenceManager


class InferenceProcessGroupTests(unittest.TestCase):
    def _manager(self):
        mgr = InferenceManager.__new__(InferenceManager)
        mgr.script_path = Path("/tmp/run_zone_detect.py")
        mgr.config_path = Path("/tmp/config.json")
        mgr.process = None
        mgr.file_source = False
        mgr.single_shot = False
        mgr.desired = False
        mgr.auto_restart = True
        mgr.auto_restart_user_set = False
        mgr.restart_count = 0
        mgr.last_start = None
        mgr.last_exit = None
        mgr.heartbeat_path = None
        mgr.startup_flag_path = None
        mgr.command_dir = None
        mgr.device_id = "RK3588-DEV"
        mgr.runtime_namespace_key = "RK3588-DEV"
        mgr.launch_id = ""
        mgr.heartbeat_timeout_seconds = 30.0
        mgr.progress_timeout_seconds = 90.0
        mgr.heartbeat_startup_grace_seconds = 90.0
        mgr.last_heartbeat_status = {}
        mgr.log_buffer = []
        mgr.log_lock = None
        mgr.shutdown = None
        mgr.lock = None
        mgr._append_log = Mock()
        mgr._load_watchdog_settings_locked = Mock()
        mgr._terminate_stale_runtime_process_locked = Mock()
        mgr._clear_heartbeat_file_locked = Mock()
        mgr._sync_source_policy_locked = Mock()
        mgr._capture_output = Mock()
        return mgr

    @patch("web.inference.threading.Thread")
    @patch("web.inference.subprocess.Popen")
    @patch("web.inference.uuid4")
    def test_launch_starts_new_session(self, uuid4_mock, popen_mock, thread_mock):
        mgr = self._manager()
        mgr.script_path = Path(__file__)
        uuid4_mock.return_value = SimpleNamespace(hex="abcdef1234567890")
        proc = Mock(pid=1234, stdout=[])
        popen_mock.return_value = proc
        thread_instance = Mock()
        thread_mock.return_value = thread_instance

        mgr._launch_locked()

        self.assertTrue(popen_mock.call_args.kwargs["start_new_session"])
        self.assertEqual(mgr.process, proc)
        thread_instance.start.assert_called_once()

    @patch("web.inference.os.killpg")
    @patch("web.inference.os.kill")
    def test_signal_target_prefers_process_group_for_group_leader(self, kill_mock, killpg_mock):
        mgr = self._manager()
        mgr._safe_getpgid = Mock(return_value=4321)

        ok = mgr._signal_target(4321, signal.SIGTERM, reason="test", allow_group=True)

        self.assertTrue(ok)
        killpg_mock.assert_called_once_with(4321, signal.SIGTERM)
        kill_mock.assert_not_called()

    @patch("web.inference.os.killpg")
    @patch("web.inference.os.kill")
    def test_signal_target_falls_back_to_pid_for_non_group_leader(self, kill_mock, killpg_mock):
        mgr = self._manager()
        mgr._safe_getpgid = Mock(return_value=9999)

        ok = mgr._signal_target(4321, signal.SIGTERM, reason="test", allow_group=True)

        self.assertTrue(ok)
        kill_mock.assert_called_once_with(4321, signal.SIGTERM)
        killpg_mock.assert_not_called()

    def test_terminate_locked_uses_group_signal_and_records_exit(self):
        mgr = self._manager()
        proc = Mock(pid=4321)
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        mgr.process = proc
        mgr._safe_getpgid = Mock(return_value=4321)
        mgr._signal_target = Mock(return_value=True)

        mgr._terminate_locked()

        mgr._signal_target.assert_called_once_with(4321, signal.SIGTERM, reason="tracked_stop", allow_group=True)
        proc.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(mgr.process)
        self.assertEqual(mgr.last_exit["code"], 0)

    def test_terminate_pid_locked_falls_back_safely_for_old_non_group_leader_process(self):
        mgr = self._manager()
        mgr.process = None
        mgr._safe_getpgid = Mock(return_value=9999)
        mgr._pid_exists = Mock(side_effect=[True, False])
        mgr._signal_target = Mock(return_value=True)

        ok = mgr._terminate_pid_locked(4321, "stale_test")

        self.assertTrue(ok)
        mgr._signal_target.assert_called_once_with(4321, signal.SIGTERM, reason="stale_term:stale_test", allow_group=True)

    def test_terminate_locked_force_kills_group_after_timeout(self):
        mgr = self._manager()
        proc = Mock(pid=4321)
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="demo", timeout=5), 0]
        proc.poll.return_value = -9
        mgr.process = proc
        mgr._safe_getpgid = Mock(return_value=4321)
        mgr._signal_target = Mock(return_value=True)

        mgr._terminate_locked()

        self.assertEqual(
            mgr._signal_target.call_args_list,
            [
                unittest.mock.call(4321, signal.SIGTERM, reason="tracked_stop", allow_group=True),
                unittest.mock.call(4321, signal.SIGKILL, reason="tracked_kill", allow_group=True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
