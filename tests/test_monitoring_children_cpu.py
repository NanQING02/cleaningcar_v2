import types
import unittest

from cleaningcar.monitoring import _snapshot_children


class _Child:
    pid = 42

    def __init__(self):
        self.total = 1.0

    def name(self):
        return 'ffmpeg'

    def cmdline(self):
        return ['ffmpeg']

    def cpu_times(self):
        return types.SimpleNamespace(user=self.total, system=0.0)

    def create_time(self):
        return 100.0

    def memory_info(self):
        return types.SimpleNamespace(rss=10 * 1024 * 1024)


class _Parent:
    def __init__(self, child):
        self.child = child

    def children(self, recursive=True):
        return [self.child]


class MonitoringChildCpuTests(unittest.TestCase):
    def test_cpu_is_computed_from_persistent_time_delta(self):
        child = _Child()
        state = {}

        first = _snapshot_children(_Parent(child), state, now=10.0)
        child.total = 1.5
        second = _snapshot_children(_Parent(child), state, now=10.5)

        self.assertEqual(first['cpu'], 0.0)
        self.assertAlmostEqual(second['cpu'], 100.0)
        self.assertEqual(second['ffmpeg_count'], 1)
        self.assertEqual(second['rss_mb'], 10.0)
