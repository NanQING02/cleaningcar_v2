import unittest

from cleaningcar.log_throttle import WindowedLogThrottler


class WindowedLogThrottlerTests(unittest.TestCase):
    def test_first_message_emits_immediately(self):
        throttler = WindowedLogThrottler(window_seconds=5.0)

        self.assertEqual(throttler.record("reader", "first", now=0.0), ["first"])
        self.assertEqual(throttler.record("reader", "second", now=1.0), [])

    def test_window_rollover_emits_summary_then_current_message(self):
        throttler = WindowedLogThrottler(window_seconds=5.0)

        throttler.record("reader", "first", now=0.0)
        throttler.record("reader", "second", now=1.0)
        throttler.record("reader", "third", now=2.0)

        messages = throttler.record("reader", "fourth", now=6.0)

        self.assertEqual(len(messages), 2)
        self.assertIn("suppressed 2 similar messages", messages[0])
        self.assertEqual(messages[1], "fourth")


if __name__ == "__main__":
    unittest.main()
