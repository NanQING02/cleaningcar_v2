import sys
import unittest
from unittest.mock import patch

from web import server


class WebServerStartupTests(unittest.TestCase):
    @patch("uvicorn.run")
    @patch("web.server.state.set_config_path")
    def test_main_disables_unused_websocket_protocol(self, set_config_path_mock, uvicorn_run_mock):
        argv = ["web.server", "--config", "/tmp/config.json"]

        with patch.object(sys, "argv", argv):
            server.main()

        set_config_path_mock.assert_called_once()
        self.assertEqual(uvicorn_run_mock.call_args.kwargs["ws"], "none")


if __name__ == "__main__":
    unittest.main()
