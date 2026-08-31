import json
import tempfile
import unittest
from pathlib import Path

from config_manager import ConfigManager
from cleaningcar.vision import scale_point, scale_polygon


class ZoneCoordinateNormalizationTests(unittest.TestCase):
    def test_scale_polygon_clamps_small_normalized_rounding_error(self):
        points = [[-0.0002, 0.5], [0.5, 1.0002], [1.0, 0.0]]

        scaled = scale_polygon(points, width=1920, height=1080)

        self.assertEqual(scaled[0], (0.0, 540.0))
        self.assertEqual(scaled[1], (960.0, 1080.0))
        self.assertEqual(scaled[2], (1920.0, 0.0))

    def test_scale_point_clamps_small_normalized_rounding_error(self):
        self.assertEqual(scale_point([-0.0002, 1.0002], 1920, 1080), (0.0, 1080.0))

    def test_scale_polygon_preserves_legacy_pixel_coordinates(self):
        points = [[10, 20], [100, 20], [100, 200]]

        self.assertEqual(scale_polygon(points, 1920, 1080), [(10.0, 20.0), (100.0, 20.0), (100.0, 200.0)])

    def test_config_manager_clamps_near_normalized_zone_and_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.json'
            payload = {
                'system': {'device_id': 'cam-a'},
                'video': {'source': 'demo.mp4'},
                'zones': {
                    'zone_a_detection': [[-0.0002, 0.5], [0.5, 1.0002], [1.0, 0.0]],
                    'zone_b_wash': [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
                    'flow_vector': {'start': [-0.0002, 0.5], 'end': [1.0002, 0.5]},
                },
            }
            path.write_text(json.dumps(payload), encoding='utf-8')

            manager = ConfigManager(path)

            self.assertEqual(manager.zones['zone_a_detection'][0], (0.0, 0.5))
            self.assertEqual(manager.zones['zone_a_detection'][1], (0.5, 1.0))
            self.assertEqual(manager.zones['flow_vector']['start'], (0.0, 0.5))
            self.assertEqual(manager.zones['flow_vector']['end'], (1.0, 0.5))


if __name__ == '__main__':
    unittest.main()
