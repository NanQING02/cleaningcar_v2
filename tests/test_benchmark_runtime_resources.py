import tempfile
import unittest
from pathlib import Path

from tools.benchmark_runtime_resources import sanitized_config


class BenchmarkRuntimeResourcesTests(unittest.TestCase):
    def test_plate_core_override_and_storage_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'out'
            output.mkdir()
            source = Path(__file__).resolve().parents[1] / 'configs' / 'config.json'

            data = sanitized_config(source, output, plate_core_mask='2', keep_wheel=True)

            self.assertEqual(data['logic']['plate_core_mask'], '2')
            self.assertTrue(data['wheel']['enabled'])
            self.assertFalse(data['logic']['enable_per_id_video'])
