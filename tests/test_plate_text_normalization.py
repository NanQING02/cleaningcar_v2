import unittest

from cleaningcar.plate import is_valid_plate, normalize_plate_candidate_text


class PlateTextNormalizationTests(unittest.TestCase):
    def test_body_confusion_is_corrected_conservatively(self):
        self.assertEqual(normalize_plate_candidate_text("鲁AOI234"), "鲁A01234")
        self.assertTrue(is_valid_plate("鲁AOI234"))

    def test_prefix_is_not_blindly_corrected(self):
        self.assertEqual(normalize_plate_candidate_text("0A12345"), "0A12345")
        self.assertFalse(is_valid_plate("0A12345"))


if __name__ == "__main__":
    unittest.main()
