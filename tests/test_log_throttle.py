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


class WindowedLogThrottlerPurgeTests(unittest.TestCase):
    def test_stale_keys_are_purged_periodically(self):
        throttler = WindowedLogThrottler(window_seconds=1.0)
        now = 1000.0
        throttler._last_purge_ts = now - 100.0  # 让下一次record触发清理

        for i in range(10):
            throttler.record(key=f'a{i}', message='m', now=now)
        self.assertEqual(len(throttler._states), 10)

        # 时间前进超过清理间隔与滞留阈值（window*4=4s），新key触发清理
        throttler.record(key='b', message='m', now=now + 100.0)
        self.assertNotIn('a0', throttler._states)
        self.assertIn('b', throttler._states)

    def test_fresh_keys_are_not_purged(self):
        throttler = WindowedLogThrottler(window_seconds=1.0)
        now = 2000.0
        throttler._last_purge_ts = now - 100.0

        for i in range(5):
            throttler.record(key=f'c{i}', message='m', now=now + i * 0.01)
        self.assertEqual(len(throttler._states), 5)
