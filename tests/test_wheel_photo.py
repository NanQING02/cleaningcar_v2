import tempfile
import unittest
from pathlib import Path

import numpy as np

from cleaningcar.events import EventManager
from cleaningcar.constants import (
    WHEEL_CLASS_NAME_TO_CLEAN_VALUE,
    WHEEL_SIDE_TO_PHOTO_TYPE,
)
from cleaningcar.wheel import WheelResultCache


class _DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx):
        return None, {"enter_a": False, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class _FakeWheelPhotoUploader:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, payload):
        self.enqueued.append(dict(payload))


class _StaticClaimer:
    def __init__(self):
        self.claims = {}

    def __call__(self, track_id, entry_id):
        track_id = int(track_id or 0)
        entry_id = int(entry_id or 0)
        if track_id <= 0 or entry_id <= 0:
            return False
        owner = self.claims.get(entry_id, 0)
        if owner and owner != track_id:
            return False
        self.claims[entry_id] = track_id
        return True


class _SingleEntryWheelProvider:
    def __init__(self, item):
        self.item = dict(item)
        self.claims = {}

    def get_recent_result_entries(self, now_ts=None, reference_ts=None, track_id=None):
        owner = int(self.claims.get(int(self.item.get('entryId', 0) or 0), 0) or 0)
        if owner > 0 and owner != int(track_id or 0):
            return []
        return [dict(self.item)]

    def claim_result_entry(self, track_id, entry_id):
        track_id = int(track_id or 0)
        entry_id = int(entry_id or 0)
        if track_id <= 0 or entry_id <= 0:
            return False
        owner = int(self.claims.get(entry_id, 0) or 0)
        if owner > 0 and owner != track_id:
            return False
        self.claims[entry_id] = track_id
        return True


class ConstantMapTests(unittest.TestCase):
    def test_class_and_side_maps(self):
        self.assertEqual(WHEEL_CLASS_NAME_TO_CLEAN_VALUE['0-25'], 1)
        self.assertEqual(WHEEL_CLASS_NAME_TO_CLEAN_VALUE['75-100'], 4)
        self.assertEqual(WHEEL_SIDE_TO_PHOTO_TYPE['left'], '4')
        self.assertEqual(WHEEL_SIDE_TO_PHOTO_TYPE['right'], '5')


class WheelPhotoTests(unittest.TestCase):
    def _manager(self, uploader=None, base_dir=None, bucket_seconds=0.25,
                 min_score=0.3, wheel_provider=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        if base_dir is None:
            base_dir = self._tmp.name
        config = {
            "logic": {
                "default_plate_color": "",
                "default_plate_color_conf": 0.0,
            },
            "wheel": {
                "bind_wait_seconds": 0.0,
            },
            "shadow_pool": {
                "max_candidates": 20,
                "max_age_frames": 30,
            },
            "event_capture_dir": self._tmp.name,
            "event_output_dir": self._tmp.name,
            "lane_name": "lane-a",
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_DummyZoneManager(),
            wheel_result_provider=wheel_provider,
            wheel_photo_uploader=uploader,
            wheel_photo_base_dir=base_dir,
            session_id='143025',
            wheel_photo_bucket_seconds=bucket_seconds,
            wheel_photo_min_score=min_score,
        )

    @staticmethod
    def _candidate(side='left', score=0.8, capture_ts=1000.0, class_name='25-50',
                   entry_id=1, image_bytes=b'fakejpg', center_distance=100.0):
        return {
            'side': side,
            'captureTime': '2026-06-20 14:30:25',
            'imageJpegBytes': image_bytes,
            'className': class_name,
            'score': float(score),
            'centerDistance': float(center_distance),
            'capture_ts': float(capture_ts),
            'entryId': int(entry_id),
        }

    def _new_track(self, mgr, track_id=1):
        return mgr.tracks.setdefault(track_id, {
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
        })

    def test_low_score_skipped(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        mgr._update_wheel_photo_history(
            track_id=1, track_state=st, side='left',
            candidate=self._candidate(score=0.2, entry_id=1),
            claimer=claimer,
        )
        self.assertEqual(st['wheel_photo_history']['left'], {})
        self.assertEqual(st['wheel_photo_seq']['left'], 0)

    def test_bucket_representative_uses_majority_type_with_low_tiebreak(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        # 同桶（capture_ts 都在 1000.0~1000.25 内）4 个候选
        # 2 张 50-75 + 1 张 0-25 + 1 张 75-100 → 多数票 50-75
        for idx, (cn, score, dist, ts) in enumerate([
            ('50-75', 0.7, 50.0, 1000.00),
            ('0-25', 0.9, 80.0, 1000.05),
            ('50-75', 0.8, 30.0, 1000.10),
            ('75-100', 0.6, 10.0, 1000.15),
        ]):
            mgr._update_wheel_photo_history(
                track_id=1, track_state=st, side='left',
                candidate=self._candidate(score=score, capture_ts=ts,
                                          class_name=cn, entry_id=idx + 1,
                                          center_distance=dist),
                claimer=claimer,
            )
        bucket = list(st['wheel_photo_history']['left'].values())[0]
        rep = bucket['representative']
        # cleanValue follows the majority type, while the photo uses the nearest box to frame center.
        self.assertEqual(rep['cleanValue'], 3)
        self.assertEqual(rep['entryId'], 4)

    def test_tie_count_prefers_lowest_type(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        # 2 张 0-25 + 2 张 75-100 → 同票，倾向类型最低 → 0-25
        for idx, (cn, score, dist, ts) in enumerate([
            ('0-25', 0.7, 50.0, 1000.00),
            ('75-100', 0.9, 80.0, 1000.05),
            ('0-25', 0.6, 30.0, 1000.10),
            ('75-100', 0.8, 10.0, 1000.15),
        ]):
            mgr._update_wheel_photo_history(
                track_id=1, track_state=st, side='left',
                candidate=self._candidate(score=score, capture_ts=ts,
                                          class_name=cn, entry_id=idx + 1,
                                          center_distance=dist),
                claimer=claimer,
            )
        bucket = list(st['wheel_photo_history']['left'].values())[0]
        rep = bucket['representative']
        self.assertEqual(rep['cleanValue'], 1)
        self.assertEqual(rep['entryId'], 4)

    def test_majority_dirty_wins_over_single_clean(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        # 1 张 0-25 vs 5 张 75-100 → 多数是脏的 → 选 75-100
        # 这就是用户说的"普遍脏才报脏的，不硬报 0-25"
        for idx, (cn, dist, ts) in enumerate([
            ('0-25', 50.0, 1000.00),
            ('75-100', 30.0, 1000.03),
            ('75-100', 40.0, 1000.06),
            ('75-100', 20.0, 1000.09),
            ('75-100', 60.0, 1000.12),
            ('75-100', 10.0, 1000.15),
        ]):
            mgr._update_wheel_photo_history(
                track_id=1, track_state=st, side='left',
                candidate=self._candidate(score=0.8, capture_ts=ts,
                                          class_name=cn, entry_id=idx + 1,
                                          center_distance=dist),
                claimer=claimer,
            )
        bucket = list(st['wheel_photo_history']['left'].values())[0]
        rep = bucket['representative']
        self.assertEqual(rep['cleanValue'], 4)
        # 75-100 里 centerDistance 最小是 10.0（entryId=6）
        self.assertEqual(rep['entryId'], 6)

    def test_different_buckets_each_keep_one_representative(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader, bucket_seconds=0.25)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        # 0, 0.3, 0.6, 1.2 秒 → 4 个不同的桶（0.25 边界）
        for idx, ts in enumerate([1000.0, 1000.3, 1000.6, 1001.2]):
            mgr._update_wheel_photo_history(
                track_id=1, track_state=st, side='left',
                candidate=self._candidate(score=0.7, capture_ts=ts,
                                          class_name='50-75', entry_id=idx + 1),
                claimer=claimer,
            )
        self.assertEqual(len(st['wheel_photo_history']['left']), 4)

    def test_side_to_type_mapping(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        mgr._update_wheel_photo_history(
            track_id=1, track_state=st, side='right',
            candidate=self._candidate(side='right', score=0.9,
                                      class_name='75-100', entry_id=1),
            claimer=claimer,
        )
        rep = list(st['wheel_photo_history']['right'].values())[0]['representative']
        self.assertEqual(rep['type'], '5')
        self.assertEqual(rep['cleanValue'], 4)

    def test_file_path_layout(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        mgr._update_wheel_photo_history(
            track_id=1, track_state=st, side='left',
            candidate=self._candidate(score=0.9, capture_ts=1718835000.0,
                                      class_name='25-50', entry_id=1),
            claimer=claimer,
        )
        rep = list(st['wheel_photo_history']['left'].values())[0]['representative']
        self.assertTrue(Path(rep['photoUrl']).is_absolute())
        self.assertTrue(rep['photoUrl'].startswith(str(mgr.wheel_photo_base_dir)))
        self.assertIn('/143025_1_left_1.jpg', rep['photoUrl'])
        self.assertTrue(Path(rep['photoUrl']).exists())
        with Path(rep['photoUrl']).open('rb') as f:
            self.assertEqual(f.read(), b'fakejpg')

    def test_representative_overwrites_same_seq_when_majority_shifts(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        # 第一帧：0-25，唯一候选 → 代表是 0-25
        mgr._update_wheel_photo_history(
            track_id=1, track_state=st, side='left',
            candidate=self._candidate(score=0.9, capture_ts=1000.0,
                                      class_name='0-25', entry_id=1,
                                      center_distance=10.0),
            claimer=claimer,
        )
        # 第二、三、四帧：都是 75-100 → 桶内变成 1 vs 3，多数票翻转
        for idx, ts in enumerate([1000.05, 1000.10, 1000.15]):
            mgr._update_wheel_photo_history(
                track_id=1, track_state=st, side='left',
                candidate=self._candidate(score=0.8, capture_ts=ts,
                                          class_name='75-100', entry_id=idx + 2,
                                          center_distance=20.0),
                claimer=claimer,
            )
        rep = list(st['wheel_photo_history']['left'].values())[0]['representative']
        self.assertEqual(rep['cleanValue'], 4)
        # 同一 seq（覆盖式落盘），seq=1
        self.assertEqual(rep['seq'], 1)

    def test_claim_failure_skips_entry(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        claimer.claims[1] = 999
        mgr._update_wheel_photo_history(
            track_id=1, track_state=st, side='left',
            candidate=self._candidate(score=0.9, entry_id=1),
            claimer=claimer,
        )
        self.assertEqual(st['wheel_photo_history']['left'], {})

    def test_enqueue_on_emit_type5(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        track_state = {
            'wheel_photo_history': {
                'left': {
                    2000: {'candidates': [], 'representative': {
                        'photoUrl': 'box/a.jpg', 'type': '4', 'cleanValue': 2,
                        'score': 0.7, 'capture_ts': 2000.0, 'entryId': 1, 'seq': 1,
                    }},
                    2010: {'candidates': [], 'representative': {
                        'photoUrl': 'box/b.jpg', 'type': '4', 'cleanValue': 3,
                        'score': 0.8, 'capture_ts': 2010.0, 'entryId': 2, 'seq': 2,
                    }},
                },
                'right': {
                    2005: {'candidates': [], 'representative': {
                        'photoUrl': 'box/c.jpg', 'type': '5', 'cleanValue': 4,
                        'score': 0.9, 'capture_ts': 2005.0, 'entryId': 3, 'seq': 1,
                    }},
                },
            },
        }
        mgr._enqueue_wheel_photos(track_state)
        self.assertEqual(len(uploader.enqueued), 3)
        ordered_urls = [p['photoUrl'] for p in uploader.enqueued]
        self.assertEqual(ordered_urls, [
            (mgr.wheel_photo_base_dir / 'box/a.jpg').resolve().as_posix(),
            (mgr.wheel_photo_base_dir / 'box/c.jpg').resolve().as_posix(),
            (mgr.wheel_photo_base_dir / 'box/b.jpg').resolve().as_posix(),
        ])
        self.assertEqual(uploader.enqueued[0],
                         {'photoUrl': (mgr.wheel_photo_base_dir / 'box/a.jpg').resolve().as_posix(),
                          'type': '4', 'cleanValue': 2})
        self.assertEqual(uploader.enqueued[2],
                         {'photoUrl': (mgr.wheel_photo_base_dir / 'box/b.jpg').resolve().as_posix(),
                          'type': '4', 'cleanValue': 3})

    def test_first_lifecycle_lock_also_records_wheel_photo(self):
        uploader = _FakeWheelPhotoUploader()
        provider = _SingleEntryWheelProvider(self._candidate(
            class_name='25-50',
            entry_id=10,
            score=0.8,
            center_distance=12.0,
        ))
        mgr = self._manager(uploader=uploader, wheel_provider=provider)
        track_state = {
            'wheel_results_locked': {},
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
            'wheel_active': True,
            'wheel_activity_start_ts': 999.0,
        }

        mgr._update_track_wheel_results(1, track_state, frame_ts=1000.0)
        mgr._enqueue_wheel_photos(track_state)

        self.assertIn('left', track_state['wheel_results_locked'])
        self.assertEqual(len(uploader.enqueued), 1)
        self.assertEqual(uploader.enqueued[0]['type'], '4')
        self.assertEqual(uploader.enqueued[0]['cleanValue'], 2)

    def test_real_cache_claimed_cluster_uses_bucket_majority(self):
        uploader = _FakeWheelPhotoUploader()
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        class_names = ['0-25', '25-50', '50-75', '75-100']
        samples = [
            (0, 0.99, 60.0, 1000.00),
            (3, 0.80, 40.0, 1000.03),
            (3, 0.80, 35.0, 1000.06),
            (3, 0.80, 20.0, 1000.09),
            (3, 0.80, 45.0, 1000.12),
        ]
        for cls_id, score, center, ts in samples:
            half = 8.0
            cache.update_from_detections(
                side='left',
                frame=frame,
                capture_ts=ts,
                boxes=np.array([[center - half, center - half, center + half, center + half]], dtype=np.float32),
                classes=np.array([cls_id], dtype=np.int64),
                scores=np.array([score], dtype=np.float32),
                class_names=class_names,
            )
        mgr = self._manager(uploader=uploader, wheel_provider=cache)
        track_state = {
            'wheel_results_locked': {},
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
            'wheel_active': True,
            'wheel_activity_start_ts': 999.0,
        }

        mgr._update_track_wheel_results(1, track_state, frame_ts=1000.12)
        mgr._enqueue_wheel_photos(track_state)

        self.assertEqual(len(uploader.enqueued), 1)
        self.assertEqual(uploader.enqueued[0]['cleanValue'], 4)
        rep = list(track_state['wheel_photo_history']['left'].values())[0]['representative']
        self.assertEqual(rep['cleanValue'], 4)

    def test_real_cache_collects_later_lower_score_bucket(self):
        uploader = _FakeWheelPhotoUploader()
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        class_names = ['0-25', '25-50', '50-75', '75-100']
        cache.update_from_detections(
            side='left',
            frame=frame,
            capture_ts=1000.0,
            boxes=np.array([[52.0, 52.0, 68.0, 68.0]], dtype=np.float32),
            classes=np.array([0], dtype=np.int64),
            scores=np.array([0.99], dtype=np.float32),
            class_names=class_names,
        )
        cache.update_from_detections(
            side='left',
            frame=frame,
            capture_ts=1002.0,
            boxes=np.array([[22.0, 22.0, 38.0, 38.0]], dtype=np.float32),
            classes=np.array([3], dtype=np.int64),
            scores=np.array([0.70], dtype=np.float32),
            class_names=class_names,
        )
        mgr = self._manager(uploader=uploader, wheel_provider=cache)
        track_state = {
            'wheel_results_locked': {},
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
            'wheel_active': True,
            'wheel_activity_start_ts': 999.0,
        }

        mgr._update_track_wheel_results(1, track_state, frame_ts=1000.0)
        mgr._update_track_wheel_results(1, track_state, frame_ts=1002.0)
        mgr._enqueue_wheel_photos(track_state)

        self.assertEqual([item['cleanValue'] for item in uploader.enqueued], [1, 4])

    def test_default_photo_bucket_seconds_is_quarter_second(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()

        for idx, ts in enumerate([1000.10, 1000.20], start=1):
            mgr._update_wheel_photo_history(
                track_id=1,
                track_state=st,
                side='left',
                candidate=self._candidate(capture_ts=ts, entry_id=idx),
                claimer=claimer,
            )

        self.assertEqual(len(st['wheel_photo_history']['left']), 1)

    def test_same_vehicle_same_side_duplicate_photo_bytes_are_skipped(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader, bucket_seconds=0.25)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()

        for idx, ts in enumerate([1000.10, 1000.60], start=1):
            mgr._update_wheel_photo_history(
                track_id=1,
                track_state=st,
                side='left',
                candidate=self._candidate(capture_ts=ts, entry_id=idx, image_bytes=b'samejpg'),
                claimer=claimer,
            )

        mgr._enqueue_wheel_photos(st)

        self.assertEqual(len(st['wheel_photo_history']['left']), 2)
        self.assertEqual(st['wheel_photo_seq']['left'], 1)
        self.assertEqual(len(uploader.enqueued), 1)

    def test_event_manager_default_photo_bucket_seconds_is_quarter_second(self):
        mgr = EventManager(
            {
                "logic": {},
                "shadow_pool": {},
                "event_capture_dir": tempfile.mkdtemp(),
                "event_output_dir": tempfile.mkdtemp(),
            },
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_DummyZoneManager(),
            wheel_photo_uploader=_FakeWheelPhotoUploader(),
        )

        self.assertEqual(mgr.wheel_photo_bucket_seconds, 0.25)

    def test_representative_uses_nearest_box_to_frame_center_across_bucket(self):
        candidates = [
            {'className': '0-25', 'centerDistance': 100.0, 'entryId': 1, 'score': 0.9},
            {'className': '50-75', 'centerDistance': 10.0, 'entryId': 2, 'score': 0.8},
            {'className': '50-75', 'centerDistance': 20.0, 'entryId': 3, 'score': 0.7},
            {'className': '75-100', 'centerDistance': 5.0, 'entryId': 4, 'score': 0.6},
        ]

        rep = EventManager._select_bucket_representative(candidates)

        self.assertEqual(rep['entryId'], 4)

    def test_select_bucket_representative_helper(self):
        candidates = [
            {'className': '0-25', 'centerDistance': 100.0, 'entryId': 1, 'score': 0.9},
            {'className': '50-75', 'centerDistance': 10.0, 'entryId': 2, 'score': 0.8},
            {'className': '50-75', 'centerDistance': 20.0, 'entryId': 3, 'score': 0.7},
            {'className': '75-100', 'centerDistance': 5.0, 'entryId': 4, 'score': 0.6},
        ]
        rep = EventManager._select_bucket_representative(candidates)
        self.assertEqual(rep['entryId'], 4)

        candidates2 = [
            {'className': '0-25', 'centerDistance': 100.0, 'entryId': 1, 'score': 0.9},
            {'className': '0-25', 'centerDistance': 50.0, 'entryId': 5, 'score': 0.9},
        ]
        rep2 = EventManager._select_bucket_representative(candidates2)
        # 全是 0-25，选 centerDistance 最小（50.0）
        self.assertEqual(rep2['entryId'], 5)


if __name__ == '__main__':
    unittest.main()
