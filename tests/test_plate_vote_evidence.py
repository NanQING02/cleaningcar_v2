import unittest

from cleaningcar.plate import PlateVoteEvidence


class PlateVoteEvidenceTests(unittest.TestCase):
    def test_sparse_source_frames_use_capture_time_not_frame_gap(self):
        votes = PlateVoteEvidence()
        for frame, ts in [(10, 100.0), (42, 100.5), (80, 101.0), (112, 101.5)]:
            votes.add(frame, ts, '苏C2267S')
        self.assertEqual(votes.majority(101.5, 2, 4, .7), ('苏C2267S', 4, 4))

    def test_same_source_frame_does_not_count_twice(self):
        votes = PlateVoteEvidence()
        for _ in range(5):
            votes.add(10, 100, '苏C2267S')
        self.assertEqual(votes.majority(100, 2, 4, .7), ('', 1, 1))

    def test_conflicting_plate_is_not_ignored_by_majority(self):
        votes = PlateVoteEvidence()
        for i, text in enumerate(['苏C2267S'] * 4 + ['苏C7755S'] * 2):
            votes.add(i, 100 + i * .1, text)
        self.assertEqual(votes.majority(100.5, 2, 4, .7), ('', 4, 6))

    def test_old_votes_expire_using_time(self):
        votes = PlateVoteEvidence()
        for i in range(3):
            votes.add(i, 100 + i * .1, '苏C2267S')
        votes.add(1000, 103, '苏C2267S')
        self.assertEqual(votes.majority(103, 2, 4, .7), ('', 1, 1))

    def test_merge_deduplicates_and_caps_memory(self):
        old = PlateVoteEvidence(max_observations=4)
        new = PlateVoteEvidence(max_observations=4)
        for i in range(3):
            old.add(i, 100 + i * .1, '苏C2267S')
        for i in range(2, 6):
            new.add(i, 100 + i * .1, '苏C2267S')
        new.merge(old, 100.5)
        self.assertEqual(len(new.observations), 4)
        self.assertEqual(len({x['frame_idx'] for x in new.observations}), 4)
        self.assertEqual(new.majority(100.5, 2, 4, .7), ('苏C2267S', 4, 4))

    def test_out_of_order_vote_is_not_added(self):
        votes = PlateVoteEvidence()
        votes.add(3, 103, '苏C2267S')
        self.assertFalse(votes.add(2, 102, '苏C2267S'))


if __name__ == '__main__':
    unittest.main()
