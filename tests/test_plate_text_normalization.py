import unittest

from cleaningcar.plate import PlateTextTracker, is_valid_plate, normalize_plate_candidate_text


class PlateTextNormalizationTests(unittest.TestCase):
    def test_tracker_locks_only_consecutive_high_confidence_text(self):
        tracker = PlateTextTracker(lock_frames=3, min_detection_confidence=0.65, min_recognition_confidence=0.75)
        detections = [
            {'box': [0, 0, 40, 15], 'text': '苏A3A329', 'score': 0.90, 'plate_text_conf': 0.90},
            {'box': [1, 0, 41, 15], 'text': '苏A3A329', 'score': 0.90, 'plate_text_conf': 0.90},
            {'box': [2, 0, 42, 15], 'text': '桂N0T7RUK', 'score': 0.90, 'plate_text_conf': 0.90},
            {'box': [3, 0, 43, 15], 'text': '苏A3A329', 'score': 0.90, 'plate_text_conf': 0.90},
            {'box': [4, 0, 44, 15], 'text': '苏A3A329', 'score': 0.90, 'plate_text_conf': 0.90},
            {'box': [5, 0, 45, 15], 'text': '苏A3A329', 'score': 0.90, 'plate_text_conf': 0.90},
        ]

        results = []
        for frame_idx, detection in enumerate(detections, start=1):
            results = tracker.update(frame_idx, [detection])

        self.assertEqual(results[0]['text'], '苏A3A329')
        self.assertFalse(results[0]['is_guess'])

    def test_tracker_never_replaces_locked_text(self):
        tracker = PlateTextTracker(lock_frames=2, min_detection_confidence=0.65, min_recognition_confidence=0.75)
        first = {'box': [0, 0, 40, 15], 'text': '苏A3A329', 'score': 0.90, 'plate_text_conf': 0.90}
        wrong = {'box': [1, 0, 41, 15], 'text': '桂N0T7RUK', 'score': 0.99, 'plate_text_conf': 0.99}

        tracker.update(1, [first])
        tracker.update(2, [first])
        results = tracker.update(3, [wrong])

        self.assertEqual(results[0]['text'], '苏A3A329')
    def test_body_confusion_is_corrected_conservatively(self):
        self.assertEqual(normalize_plate_candidate_text("鲁AOI234"), "鲁A01234")
        self.assertTrue(is_valid_plate("鲁AOI234"))

    def test_prefix_is_not_blindly_corrected(self):
        self.assertEqual(normalize_plate_candidate_text("0A12345"), "0A12345")
        self.assertFalse(is_valid_plate("0A12345"))

    def test_prefix_i_and_o_are_not_valid_plate_letters(self):
        self.assertFalse(is_valid_plate("京I12345"))
        self.assertFalse(is_valid_plate("京O12345"))

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
