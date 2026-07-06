import unittest

from cleaningcar.wheel_remote import RemoteWheelResultProvider


class RemoteWheelResultProviderTests(unittest.TestCase):
    def test_unavailable_service_degrades_to_empty_results(self):
        provider = RemoteWheelResultProvider(
            "http://127.0.0.1:9",
            camera_id="test-camera",
            timeout=0.1,
        )

        provider.update_track_activity(1, frame_ts=1000.0, active=True)

        self.assertEqual(provider.snapshot_stats(), {})
        self.assertEqual(provider.get_recent_result_entries(track_id=1), [])
        self.assertEqual(provider.get_photo_candidate_entries(track_id=1), [])
        self.assertEqual(provider.get_claimed_result_entries(track_id=1), [])
        self.assertFalse(provider.claim_result_entry(1, 1))


if __name__ == "__main__":
    unittest.main()
