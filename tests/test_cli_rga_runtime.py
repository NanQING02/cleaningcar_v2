import os
import unittest
from unittest.mock import patch

from cleaningcar import cli


class CliRgaRuntimeTests(unittest.TestCase):
    def test_configure_rga_environment_disables_rga_by_default(self):
        config = {"video": {"rga_enable": False}}
        env = {}

        with patch.dict(os.environ, env, clear=True):
            cli._configure_rga_environment(config)
            self.assertEqual(os.environ.get("CLEANINGCAR_RGA_DISABLE"), "1")

    def test_configure_rga_environment_still_disables_rga_when_config_requests_it(self):
        config = {"video": {"rga_enable": True}}

        with patch.dict(os.environ, {}, clear=True):
            cli._configure_rga_environment(config)
            self.assertEqual(os.environ.get("CLEANINGCAR_RGA_DISABLE"), "1")

    def test_explicit_environment_enable_is_ignored(self):
        config = {"video": {"rga_enable": False}}

        with patch.dict(os.environ, {"CLEANINGCAR_RGA_ENABLE": "1"}, clear=True):
            cli._configure_rga_environment(config)
            self.assertEqual(os.environ.get("CLEANINGCAR_RGA_DISABLE"), "1")


if __name__ == "__main__":
    unittest.main()
