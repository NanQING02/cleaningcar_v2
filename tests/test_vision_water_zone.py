import unittest

from cleaningcar.vision import box_intersects_polygon, point_in_polygon


class WaterZoneGeometryTests(unittest.TestCase):
    def setUp(self):
        self.zone = [(100, 100), (300, 100), (300, 300), (100, 300)]

    def test_point_in_polygon_includes_boundary(self):
        self.assertTrue(point_in_polygon((200, 200), self.zone))
        self.assertTrue(point_in_polygon((100, 180), self.zone))
        self.assertFalse(point_in_polygon((50, 50), self.zone))

    def test_box_inside_or_crossing_zone_is_accepted(self):
        self.assertTrue(box_intersects_polygon((150, 150, 220, 220), self.zone))
        self.assertTrue(box_intersects_polygon((50, 180, 350, 220), self.zone))
        self.assertTrue(box_intersects_polygon((90, 90, 110, 110), self.zone))

    def test_box_outside_zone_is_rejected(self):
        self.assertFalse(box_intersects_polygon((10, 10, 50, 50), self.zone))


if __name__ == '__main__':
    unittest.main()
