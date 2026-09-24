import unittest

from cleaningcar.watchdog import ResultWatchdog


class ResultWatchdogTests(unittest.TestCase):
    def test_consecutive_timeouts_abort_at_limit(self):
        watchdog = ResultWatchdog(max_consecutive=3)

        self.assertFalse(watchdog.record_timeout())
        self.assertFalse(watchdog.record_timeout())
        self.assertTrue(watchdog.record_timeout())
        self.assertEqual(watchdog.total, 3)

    def test_real_result_resets_consecutive_timeout_count(self):
        watchdog = ResultWatchdog(max_consecutive=2)

        self.assertFalse(watchdog.record_timeout())
        watchdog.record_success()
        self.assertFalse(watchdog.record_timeout())
        self.assertEqual(watchdog.consecutive, 1)
        self.assertEqual(watchdog.total, 2)


if __name__ == '__main__':
    unittest.main()
