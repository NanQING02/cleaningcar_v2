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

    def test_finalize_retains_sparse_votes_after_rolling_window_expires(self):
        votes = PlateVoteEvidence()
        for frame, ts in [(1, 100), (10, 104), (20, 112), (21, 112.2)]:
            votes.add(frame, ts, '苏CJ61D6')
        self.assertEqual(votes.majority(112.2, 2, 4, .7)[0], '')
        self.assertEqual(votes.finalize_candidate()[0], '苏CJ61D6')

    def test_finalize_accepts_clear_winner_despite_scattered_errors(self):
        votes = PlateVoteEvidence()
        for i, text in enumerate(['苏C03000F'] * 9 + ['苏C03D00F'] * 2 + ['湘C03000F']):
            votes.add(i, 100 + i, text)
        self.assertEqual(votes.finalize_candidate()[0], '苏C03000F')

    def test_finalize_rejects_single_vote_and_near_tied_candidates(self):
        votes = PlateVoteEvidence()
        votes.add(1, 100, '苏CGLA67')
        self.assertEqual(votes.finalize_candidate()[1]['reason'], 'insufficient_votes')
        other = PlateVoteEvidence()
        for i, text in enumerate(['苏C07272F'] * 11 + ['苏CD7272F'] * 7):
            other.add(i, 100 + i, text)
        self.assertEqual(other.finalize_candidate()[1]['reason'], 'competing_votes')

    def test_finalize_rejects_distinct_identity_even_when_count_ratio_passes(self):
        votes = PlateVoteEvidence()
        for i, text in enumerate(['苏C02502D'] * 10 + ['苏C0699S'] * 3):
            votes.add(i, 100 + i, text)
        self.assertEqual(votes.finalize_candidate()[1]['reason'], 'distinct_plate_identity_conflict')

    def test_candidate_overflow_never_discards_rivals_to_create_winner(self):
        votes = PlateVoteEvidence(lifetime_max_candidates=2)
        for i, text in enumerate(['苏C03000F'] * 4 + ['苏C03D00F', '湘C03000F']):
            votes.add(i, 100 + i, text)
        self.assertLessEqual(len(votes.lifetime_counts), 2)
        self.assertEqual(votes.finalize_candidate()[1]['reason'], 'ambiguous_or_overflow')

    def test_duplicate_frame_after_expiry_does_not_add_lifetime_vote(self):
        votes = PlateVoteEvidence()
        votes.add(1, 100, '苏C03000F')
        votes.expire(120)
        self.assertFalse(votes.add(1, 100, '苏C03000F'))
        self.assertEqual(votes.lifetime_counts['苏C03000F'], 1)

    def test_nonoverlapping_handoff_merges_lifetime_counts(self):
        old = PlateVoteEvidence(); new = PlateVoteEvidence()
        for frame in (1, 2):old.add(frame, 100 + frame, '苏C03000F')
        for frame in (10, 11):new.add(frame, 100 + frame, '苏C03000F')
        new.merge(old, 111)
        self.assertEqual(new.finalize_candidate()[0], '苏C03000F')

    def test_overlapping_handoff_cannot_double_count_into_finalize_lock(self):
        old = PlateVoteEvidence(); new = PlateVoteEvidence()
        for frame in (1, 2):old.add(frame, 100 + frame, '苏C03000F')
        for frame in (2, 3):new.add(frame, 100 + frame, '苏C03000F')
        new.merge(old, 103)
        self.assertEqual(new.finalize_candidate()[1]['reason'], 'ambiguous_or_overflow')


if __name__ == '__main__':
    unittest.main()
