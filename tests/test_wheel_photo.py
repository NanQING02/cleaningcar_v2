import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np

from cleaningcar.events import EventManager
from cleaningcar.constants import (
    WHEEL_CLASS_NAME_TO_CLEAN_VALUE,
    WHEEL_SIDE_TO_PHOTO_TYPE,
)
from cleaningcar.wheel import WheelResultCache, _crop_wheel_photo


class _DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
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
        self.released_photo_entries = []

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

    def release_result_entry_photo(self, entry_id):
        entry_id = int(entry_id or 0)
        if entry_id != int(self.item.get('entryId', 0) or 0):
            return False
        self.item['imageJpegBytes'] = b''
        self.released_photo_entries.append(entry_id)
        return True


class _ReplayClaimedWheelProvider:
    """模拟provider每帧重放30秒窗口内全部claimed entries。"""

    def __init__(self, items):
        self.items = [dict(item) for item in items]

    def get_photo_candidate_entries(self, track_id, now_ts=None, reference_ts=None):
        return [dict(item) for item in self.items]

    def get_claimed_result_entries(self, track_id, now_ts=None, reference_ts=None):
        return [dict(item) for item in self.items]

    @staticmethod
    def claim_result_entry(track_id, entry_id):
        return int(track_id or 0) > 0 and int(entry_id or 0) > 0


class ConstantMapTests(unittest.TestCase):
    def test_class_and_side_maps(self):
        self.assertEqual(WHEEL_CLASS_NAME_TO_CLEAN_VALUE['0-25'], 1)
        self.assertEqual(WHEEL_CLASS_NAME_TO_CLEAN_VALUE['75-100'], 4)
        self.assertEqual(WHEEL_SIDE_TO_PHOTO_TYPE['left'], '4')
        self.assertEqual(WHEEL_SIDE_TO_PHOTO_TYPE['right'], '5')


class WheelPhotoTests(unittest.TestCase):
    def _manager(self, uploader=None, base_dir=None, bucket_seconds=0.5,
                 min_score=0.3, wheel_provider=None, history_max_buckets=20,
                 persistence_mode='bucket_stream', max_persisted_per_side=10):
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
            wheel_photo_history_max_buckets=history_max_buckets,
            wheel_photo_persistence_mode=persistence_mode,
            wheel_photo_max_persisted_per_side=max_persisted_per_side,
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
        # 同桶（capture_ts 都在 1000.0~1000.5 内）4 个候选
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
        Path(rep['photoUrl']).resolve().relative_to(mgr.wheel_photo_base_dir.resolve())
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
        self.assertEqual(provider.released_photo_entries, [10])
        self.assertEqual(provider.item['imageJpegBytes'], b'')

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

    def test_provider_can_release_photo_bytes_without_losing_result_metadata(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        frame = np.full((96, 96, 3), 127, dtype=np.uint8)
        self.assertTrue(cache.update_from_detections(
            side='left',
            frame=frame,
            capture_ts=1000.0,
            boxes=np.array([[10, 10, 70, 70]], dtype=np.float32),
            classes=np.array([1], dtype=np.int64),
            scores=np.array([0.9], dtype=np.float32),
        ))
        entry = cache._entries['left'][0]
        self.assertTrue(entry['imageJpegBytes'])

        self.assertTrue(cache.release_entry_photo(entry['entryId']))

        self.assertEqual(entry['imageJpegBytes'], b'')
        self.assertEqual(entry['className'], '25-50')
        self.assertEqual(entry['capture_ts'], 1000.0)
        self.assertEqual(entry['claimedTrackId'], 0)

    def test_default_photo_bucket_seconds_is_half_second(self):
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

    def test_realtime_enqueue_only_stable_buckets_and_skips_duplicates(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader, bucket_seconds=0.5)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()

        for idx, ts in enumerate([1000.10, 1000.60], start=1):
            mgr._update_wheel_photo_history(
                track_id=1,
                track_state=st,
                side='left',
                candidate=self._candidate(
                    capture_ts=ts,
                    entry_id=idx,
                    class_name='25-50',
                    image_bytes=f'jpg-{idx}'.encode('ascii'),
                ),
                claimer=claimer,
            )

        mgr._enqueue_wheel_photos(st, now_ts=1000.75, force=False)
        self.assertEqual(len(uploader.enqueued), 1)

        mgr._enqueue_wheel_photos(st, now_ts=1001.20, force=False)
        self.assertEqual(len(uploader.enqueued), 2)

        mgr._enqueue_wheel_photos(st, now_ts=1002.00, force=True, track_id=1)
        self.assertEqual(len(uploader.enqueued), 2)

    def test_candidate_photos_are_stripped_after_representative_saved(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        for idx, ts in enumerate([1000.00, 1000.05, 1000.10], start=1):
            mgr._update_wheel_photo_history(
                track_id=1,
                track_state=st,
                side='left',
                candidate=self._candidate(capture_ts=ts, entry_id=idx,
                                          image_bytes=f'jpg-{idx}'.encode('ascii')),
                claimer=claimer,
            )
        bucket = list(st['wheel_photo_history']['left'].values())[0]
        self.assertIsNotNone(bucket.get('representative'))
        # 代表已落盘后，候选只留元数据，不再持有JPEG字节
        for candidate in bucket['candidates']:
            self.assertEqual(candidate.get('imageJpegBytes'), b'')

    def test_photo_history_buckets_are_capped(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader, bucket_seconds=0.25,
                            history_max_buckets=3)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        # 5个不同桶（每0.25s一个），上限3 → 只保留最新3个
        for idx, ts in enumerate([1000.0, 1000.3, 1000.6, 1000.9, 1001.2], start=1):
            mgr._update_wheel_photo_history(
                track_id=1,
                track_state=st,
                side='left',
                candidate=self._candidate(capture_ts=ts, entry_id=idx),
                claimer=claimer,
            )
        side_history = st['wheel_photo_history']['left']
        self.assertEqual(len(side_history), 3)
        self.assertEqual(sorted(side_history.keys()), [4002, 4003, 4004])

    def test_pruned_provider_entries_are_not_rehydrated_and_resaved(self):
        uploader = _FakeWheelPhotoUploader()
        items = [
            self._candidate(
                capture_ts=1000.0 + 0.5 * idx,
                entry_id=idx + 1,
                center_distance=100.0,
                image_bytes=f'jpg-{idx + 1}'.encode('ascii'),
            )
            for idx in range(60)
        ]
        provider = _ReplayClaimedWheelProvider(items)
        mgr = self._manager(
            uploader=uploader,
            wheel_provider=provider,
            bucket_seconds=0.5,
            history_max_buckets=3,
            max_persisted_per_side=100,
        )
        st = self._new_track(mgr)

        with unittest.mock.patch.object(
                mgr, '_save_wheel_photo', wraps=mgr._save_wheel_photo) as save_mock:
            mgr._update_track_wheel_photo_history_from_provider(
                track_id=1,
                track_state=st,
                ref_ts=1030.0,
            )
            first_save_count = save_mock.call_count
            for _ in range(20):
                mgr._update_track_wheel_photo_history_from_provider(
                    track_id=1,
                    track_state=st,
                    ref_ts=1030.0,
                )

        # 60个entry只处理一次；history裁到3桶后，provider重放不会让旧桶复活。
        self.assertEqual(first_save_count, 60)
        self.assertEqual(save_mock.call_count, first_save_count)
        self.assertEqual(len(st['wheel_photo_history']['left']), 3)
        self.assertEqual(len(st['_wheel_photo_processed_entries']['left']), 60)
        self.assertEqual(st['wheel_photo_seq']['left'], 60)

    def test_bucket_stream_persists_at_most_configured_photos_per_side(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(
            uploader=uploader,
            bucket_seconds=0.5,
            history_max_buckets=20,
            max_persisted_per_side=10,
        )
        st = self._new_track(mgr)

        for side in ('left', 'right'):
            for idx in range(12):
                mgr._update_wheel_photo_history(
                    track_id=1,
                    track_state=st,
                    side=side,
                    candidate=self._candidate(
                        side=side,
                        capture_ts=1000.0 + idx,
                        entry_id=idx + 1,
                        image_bytes=f'{side}-{idx}'.encode('ascii'),
                    ),
                )

        # 过程图每侧最多9张，给type5最终锁定图预留第10个名额。
        self.assertEqual(st['wheel_photo_seq']['left'], 9)
        self.assertEqual(st['wheel_photo_seq']['right'], 9)
        self.assertEqual(len(list(Path(mgr.wheel_photo_base_dir).rglob('*.jpg'))), 18)

        st['wheel_results_locked'] = {
            side: self._candidate(
                side=side,
                capture_ts=2000.0,
                entry_id=100 + index,
                image_bytes=f'{side}-final'.encode('ascii'),
            )
            for index, side in enumerate(('left', 'right'))
        }
        mgr._enqueue_wheel_photos(st, now_ts=2001.0, force=True, track_id=1)
        wheel_results = mgr._build_wheel_results_payload(st, track_id=1)

        self.assertEqual(st['wheel_photo_seq']['left'], 10)
        self.assertEqual(st['wheel_photo_seq']['right'], 10)
        self.assertEqual(len(list(Path(mgr.wheel_photo_base_dir).rglob('*.jpg'))), 20)
        self.assertEqual(len(uploader.enqueued), 20)
        self.assertEqual(len(wheel_results), 2)
        self.assertTrue(all(Path(item['photoUrl']).exists() for item in wheel_results))

    def test_failed_photo_save_keeps_best_candidate_bytes_for_retry(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader)
        st = self._new_track(mgr)
        claimer = _StaticClaimer()
        with unittest.mock.patch.object(
                EventManager, '_save_wheel_photo', return_value=''):
            mgr._update_wheel_photo_history(
                track_id=1,
                track_state=st,
                side='left',
                candidate=self._candidate(capture_ts=1000.0, entry_id=1,
                                          image_bytes=b'jpg-1'),
                claimer=claimer,
            )
        bucket = list(st['wheel_photo_history']['left'].values())[0]
        # 保存失败时保留最佳候选字节用于重试
        kept = [c for c in bucket['candidates'] if c.get('imageJpegBytes')]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['entryId'], 1)

        # 下一次更优候选到来时应能成功保存并成为代表
        mgr._update_wheel_photo_history(
            track_id=1,
            track_state=st,
            side='left',
            candidate=self._candidate(capture_ts=1000.05, entry_id=2,
                                      image_bytes=b'jpg-2', center_distance=1.0),
            claimer=claimer,
        )
        rep = bucket.get('representative')
        self.assertIsNotNone(rep)
        self.assertEqual(rep['entryId'], 2)

    def test_wheel_photo_crop_reduces_jpeg_size(self):
        cache = WheelResultCache(bind_window_seconds=30.0, image_quality=80)
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        frame[500:560, 900:960] = 255
        cache.update_from_detections(
            side='left',
            frame=frame,
            capture_ts=1000.0,
            boxes=np.array([[900.0, 500.0, 960.0, 560.0]], dtype=np.float32),
            classes=np.array([0], dtype=np.int64),
            scores=np.array([0.9], dtype=np.float32),
            class_names=['0-25', '25-50', '50-75', '75-100'],
        )
        entries = cache._entries.get('left') or []
        self.assertEqual(len(entries), 1)
        cropped_bytes = entries[0]['imageJpegBytes']
        self.assertTrue(cropped_bytes)

        ok, full_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        self.assertTrue(ok)
        # 特写远小于整帧编码
        self.assertLess(len(cropped_bytes), len(full_encoded.tobytes()) // 4)

    def test_type5_force_enqueue_includes_locked_photo_not_in_history(self):
        uploader = _FakeWheelPhotoUploader()
        mgr = self._manager(uploader=uploader, bucket_seconds=0.5)
        st = self._new_track(mgr)
        st['wheel_results_locked'] = {
            'right': self._candidate(
                side='right',
                capture_ts=1718835001.0,
                class_name='75-100',
                entry_id=9,
                image_bytes=b'lockedjpg',
            )
        }

        mgr._enqueue_wheel_photos(st, now_ts=1718835001.0, force=True, track_id=1)

        self.assertEqual(len(uploader.enqueued), 1)
        self.assertEqual(uploader.enqueued[0]['type'], '5')
        self.assertEqual(uploader.enqueued[0]['cleanValue'], 4)
        self.assertTrue(Path(uploader.enqueued[0]['photoUrl']).exists())

    def test_final_locked_mode_only_persists_one_photo_at_type5(self):
        uploader = _FakeWheelPhotoUploader()
        candidate = self._candidate(
            side='left',
            capture_ts=1718835001.0,
            class_name='75-100',
            entry_id=9,
            image_bytes=b'lockedjpg',
        )
        provider = _SingleEntryWheelProvider(candidate)
        mgr = self._manager(
            uploader=uploader,
            wheel_provider=provider,
            persistence_mode='final_locked',
        )
        st = self._new_track(mgr)
        st['wheel_active'] = True
        st['wheel_activity_start_ts'] = 1718835000.0

        for _ in range(20):
            mgr._update_track_wheel_results(1, st, frame_ts=1718835001.0)

        self.assertIn('left', st['wheel_results_locked'])
        self.assertEqual(st['wheel_photo_history']['left'], {})
        self.assertEqual(uploader.enqueued, [])
        self.assertEqual(list(Path(mgr.wheel_photo_base_dir).rglob('*.jpg')), [])

        mgr._enqueue_wheel_photos(st, now_ts=1718835002.0, force=True, track_id=1)
        mgr._enqueue_wheel_photos(st, now_ts=1718835003.0, force=True, track_id=1)

        self.assertEqual(len(uploader.enqueued), 1)
        self.assertEqual(uploader.enqueued[0]['cleanValue'], 4)
        self.assertEqual(len(list(Path(mgr.wheel_photo_base_dir).rglob('*.jpg'))), 1)

    def test_event_manager_default_photo_bucket_seconds_is_half_second(self):
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

        self.assertEqual(mgr.wheel_photo_bucket_seconds, 0.5)
        self.assertEqual(mgr.wheel_photo_persistence_mode, 'bucket_stream')
        self.assertEqual(mgr.wheel_photo_max_persisted_per_side, 50)

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


class WheelPhotoCropTests(unittest.TestCase):
    def _frame(self, height=100, width=100):
        return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)

    def test_crop_expands_by_margin_and_clamps_to_frame(self):
        frame = self._frame()
        crop = _crop_wheel_photo(frame, [40, 40, 60, 60], margin_ratio=0.5)
        # box 20x20，四周各扩10 → 40x40
        self.assertEqual(crop.shape, (40, 40, 3))
        np.testing.assert_array_equal(crop, frame[30:70, 30:70])

        # 靠近边缘时被画面边界截断
        crop_edge = _crop_wheel_photo(frame, [0, 0, 20, 20], margin_ratio=0.5, min_size=16)
        self.assertEqual(crop_edge.shape, (30, 30, 3))
        np.testing.assert_array_equal(crop_edge, frame[0:30, 0:30])

    def test_negative_margin_ratio_returns_full_frame(self):
        frame = self._frame()
        crop = _crop_wheel_photo(frame, [40, 40, 60, 60], margin_ratio=-1.0)
        self.assertIs(crop, frame)

    def test_zero_margin_crops_exact_box(self):
        frame = self._frame()
        crop = _crop_wheel_photo(frame, [10, 10, 50, 40], margin_ratio=0.0, min_size=16)
        np.testing.assert_array_equal(crop, frame[10:40, 10:50])

    def test_crop_below_min_size_returns_full_frame(self):
        frame = self._frame()
        crop = _crop_wheel_photo(frame, [0, 0, 20, 20], margin_ratio=0.5)
        self.assertIs(crop, frame)

    def test_invalid_box_returns_full_frame(self):
        frame = self._frame()
        for box in (None, [1, 2, 3], [60, 60, 40, 40]):
            crop = _crop_wheel_photo(frame, box, margin_ratio=0.5)
            self.assertIs(crop, frame)


if __name__ == '__main__':
    unittest.main()
