import unittest

from cleaningcar.plate import is_valid_plate, normalize_plate_candidate_text


class PlateTextNormalizationTests(unittest.TestCase):
    def test_body_confusion_is_corrected_conservatively(self):
        self.assertEqual(normalize_plate_candidate_text("鲁AOI234"), "鲁A01234")
        self.assertTrue(is_valid_plate("鲁AOI234"))

    def test_prefix_is_not_blindly_corrected(self):
        self.assertEqual(normalize_plate_candidate_text("0A12345"), "0A12345")
        self.assertFalse(is_valid_plate("0A12345"))

    def test_all_letter_serial_is_rejected_as_obvious_ocr_noise(self):
        self.assertEqual(normalize_plate_candidate_text("吉MEJWUN"), "吉MEJWUN")
        self.assertFalse(is_valid_plate("吉MEJWUN"))
        self.assertFalse(is_valid_plate("京AABCDE"))

    def test_all_letter_serial_is_not_synthesized_from_confusion_chars(self):
        self.assertEqual(normalize_plate_candidate_text("吉MEOIUN"), "吉MEOIUN")
        self.assertFalse(is_valid_plate("吉MEOIUN"))

    def test_standard_and_new_energy_plates_with_digits_remain_valid(self):
        self.assertTrue(is_valid_plate("京AAB1DE"))
        self.assertTrue(is_valid_plate("粤BDF1234"))


if __name__ == "__main__":
    unittest.main()
