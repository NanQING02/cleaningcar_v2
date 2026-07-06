import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from cleaningcar import perf_lock


class PerfLockTests(unittest.TestCase):
    def test_perf_lock_enabled_defaults_to_true(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(perf_lock.perf_lock_enabled({}))

    def test_perf_lock_enabled_prefers_env_override(self):
        config = {"system": {"performance_lock_enabled": False}}
        with patch.dict("os.environ", {"CLEANINGCAR_PERF_LOCK": "1"}, clear=True):
            self.assertTrue(perf_lock.perf_lock_enabled(config))

    def test_perf_lock_restore_enabled_reads_config(self):
        config = {"system": {"performance_lock_restore_on_stop": True}}
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(perf_lock.perf_lock_restore_enabled(config))

    def test_build_command_uses_sudo_for_non_root_when_available(self):
        with patch.object(perf_lock.os, "geteuid", return_value=1000), \
                patch.object(perf_lock.shutil, "which", return_value="/usr/bin/sudo"), \
                patch.dict("os.environ", {}, clear=True):
            cmd = perf_lock.build_perf_lock_command(Path("/tmp/freq.sh"), "apply")
        self.assertEqual(cmd, ["sudo", "-n", "bash", "/tmp/freq.sh", "apply"])

    def test_build_command_skips_sudo_when_env_explicitly_disables_it(self):
        with patch.object(perf_lock.os, "geteuid", return_value=1000), \
                patch.object(perf_lock.shutil, "which", return_value="/usr/bin/sudo"), \
                patch.dict("os.environ", {"CLEANINGCAR_PERF_LOCK_USE_SUDO": "0"}, clear=True):
            cmd = perf_lock.build_perf_lock_command(Path("/tmp/freq.sh"), "apply")
        self.assertEqual(cmd, ["bash", "/tmp/freq.sh", "apply"])

    def test_build_command_skips_sudo_when_root(self):
        with patch.object(perf_lock.os, "geteuid", return_value=0):
            cmd = perf_lock.build_perf_lock_command(Path("/tmp/freq.sh"), "status")
        self.assertEqual(cmd, ["bash", "/tmp/freq.sh", "status"])

    def test_maybe_apply_perf_lock_noop_when_disabled(self):
        config = {"system": {"performance_lock_enabled": False}}
        with patch.dict("os.environ", {}, clear=True):
            result = perf_lock.maybe_apply_perf_lock(config)
        self.assertFalse(result["attempted"])
        self.assertEqual(result["reason"], "disabled")

    def test_run_perf_lock_action_reports_missing_script(self):
        config = {"system": {"performance_lock_enabled": True, "performance_lock_script": "/tmp/not-found.sh"}}
        with patch.dict("os.environ", {}, clear=True):
            result = perf_lock.run_perf_lock_action("apply", config=config)
        self.assertFalse(result["attempted"])
        self.assertEqual(result["reason"], "missing_script")

    def test_maybe_apply_perf_lock_executes_script_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "freq.sh"
            script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            config = {
                "system": {
                    "performance_lock_enabled": True,
                    "performance_lock_script": str(script_path),
                }
            }
            with patch.dict("os.environ", {}, clear=True), \
                    patch.object(perf_lock.os, "geteuid", return_value=0), \
                    patch.object(
                        perf_lock.subprocess,
                        "run",
                        return_value=CompletedProcess(
                            args=["bash", str(script_path), "apply"],
                            returncode=0,
                            stdout="[perf-lock] ok\n",
                            stderr="",
                        ),
                    ) as mock_run:
                result = perf_lock.maybe_apply_perf_lock(config)
        self.assertTrue(result["attempted"])
        self.assertTrue(result["ok"])
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
