import json
import tempfile
import unittest
from pathlib import Path

from cleaningcar.event_trace import EventTraceRecorder
from cleaningcar.events import EventManager, _format_track_debug_text
from cleaningcar.tracking import VehicleTracker, resolve_track_retention_frames
from cleaningcar.video_io import emit_per_id_video_type6


class _ScriptedZoneManager:
    def update_track(self, track_id, anchor_point, frame_idx, vehicle_height=None):
        del track_id, anchor_point
        inside_b = frame_idx >= 2
        state = type('ZoneState', (), {'inside_a': True, 'inside_b': inside_b})()
        return state, {
            'enter_a': frame_idx == 1,
            'exit_a': False,
            'enter_b': frame_idx == 2,
            'exit_b': False,
        }

    @staticmethod
    def drop_track(track_id):
        del track_id

    @staticmethod
    def resolve_direction(state):
        del state
        return 0, ''


class _NoEventZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
        del track_id, anchor_point, frame_idx, vehicle_height
        return None, {'enter_a': False, 'exit_a': False, 'enter_b': False, 'exit_b': False}

    @staticmethod
    def drop_track(track_id):
        del track_id

    @staticmethod
    def resolve_direction(state):
        del state
        return 0, ''

    @staticmethod
    def _relative_position(point):
        del point
        return 0.0


class _BornInsideZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
        del track_id, anchor_point, vehicle_height
        state = type(
            'ZoneState',
            (),
            {
                'inside_a': True,
                'inside_b': True,
                'born_inside_a': True,
                'born_inside_pending': False,
                'born_inside_outcome': 'PROMOTED',
            },
        )()
        return state, {
            'enter_a': frame_idx == 1,
            'exit_a': False,
            'enter_b': frame_idx == 1,
            'exit_b': False,
        }

    @staticmethod
    def drop_track(track_id):
        del track_id

    @staticmethod
    def resolve_direction(state):
        del state
        return 0, ''

    @staticmethod
    def _relative_position(point):
        del point
        return 0.0


class _CollectingUploader:
    def __init__(self):
        self.payloads = []

    def enqueue(self, payload):
        self.payloads.append(dict(payload))


class M1EventHardeningTests(unittest.TestCase):
    def test_debug_track_text_does_not_require_removed_stationary_fields(self):
        text = _format_track_debug_text(7, {
            'state': 'washing',
            'water': True,
            'water_consecutive_frames': 3,
            'wash_duration': 1.25,
            'zone_b_elapsed': 14,
        })
        sparse_text = _format_track_debug_text(8, {'state': 'idle'})

        self.assertEqual(text, 'ID:7 washing water:Y wf:3 dur:1.2 zb:14')
        self.assertEqual(sparse_text, 'ID:8 idle water:N wf:0 dur:0.0 zb:0')

    def test_track_retention_uses_source_fps(self):
        policy = resolve_track_retention_frames(
            25.0,
            logic_cfg={'track_lost_grace_seconds': 8.0},
            config={'track_max_age': 100, 'track_timeout_frames': 80},
        )

        self.assertEqual(policy['tracker_max_age_frames'], 200)
        self.assertEqual(policy['event_timeout_frames'], 225)

    def test_bytetrack_exposes_lost_id_until_max_age(self):
        tracker = VehicleTracker(iou_thresh=0.1, max_age=200, center_gate_ratio=0.0)
        track_id = tracker.update(1, [{'box': [0, 0, 20, 20], 'score': 0.95, 'cls': 0}])[0]

        tracker.update(100, [])
        self.assertIn(track_id, tracker.get_retained_track_ids())

        tracker.update(202, [])
        self.assertNotIn(track_id, tracker.get_retained_track_ids())

    def test_flush_inactive_keeps_tracker_retained_event_state(self):
        manager = self._manager()
        manager.tracks[1] = {'last_frame_idx': 1, 'events': set()}

        manager.flush_inactive(active_ids={1}, frame_idx=manager.timeout_frames + 100)

        self.assertIn(1, manager.tracks)

    def test_type2_backfills_type1_and_flushes_in_order(self):
        uploader = _CollectingUploader()
        manager = self._manager(uploader=uploader)
        manager._save_event_capture = lambda *args, **kwargs: ''

        self._update(manager, frame_idx=1)
        self.assertEqual(uploader.payloads, [])

        self._update(manager, frame_idx=2)

        self.assertEqual([payload['type'] for payload in uploader.payloads], [1, 2])
        self.assertEqual(uploader.payloads[0]['id'], uploader.payloads[1]['id'])
        self.assertEqual(manager.tracks[1]['events'], {1, 2})

    def test_event_stage_regression_is_suppressed_and_recorded(self):
        manager = self._manager()
        track_state = {
            'event_stage_max': 4,
            'event_sequence_issues': [],
            'abnormal_reasons': set(),
        }

        allowed = manager._event_stage_allowed(1, 3, 20, track_state)

        self.assertFalse(allowed)
        self.assertEqual(track_state['event_sequence_issues'][0]['previousStage'], 4)
        self.assertIn('OUT_OF_ORDER_TYPE3_AFTER_TYPE4', track_state['abnormal_reasons'])

    def test_inactive_flush_emits_type5_before_type6_callback(self):
        manager = self._manager()
        emitted = []
        track_state = {
            'events': {1, 2, 4},
            'event_stage_max': 4,
            'event_sequence_issues': [],
            'abnormal_reasons': set(),
            'last_frame_idx': 10,
            'last_frame': None,
            'last_vehicle_box': [0, 0, 40, 40],
            'vehicle_hit_frames': 20,
            'zone_a_seen': True,
            'zone_a_dwell_frames': 30,
            'type2_qualified': True,
            'record_start_frame': 1,
            'record_stop_frame': None,
            'vehicle_cls': 'car',
            'vehicle_cls_locked': 'car',
        }
        manager.tracks[1] = track_state

        def fake_emit_core(track_id, event_type, frame_idx, frame, payload, state, vehicle_type):
            del track_id, frame_idx, frame, payload, vehicle_type
            if manager._event_stage_allowed(1, event_type, 10, state):
                emitted.append(event_type)
                return True
            return False

        manager._emit_event_core = fake_emit_core

        manager.flush_inactive(
            active_ids=set(),
            frame_idx=manager.timeout_frames + 20,
            on_track_timeout=lambda track_id, state: emit_per_id_video_type6(
                track_id,
                state,
                manager,
                per_id_video_enabled=False,
            ),
        )

        self.assertEqual(emitted, [5, 6])

    def test_inactive_timeout_uses_capture_time_grace_window(self):
        manager = self._manager()
        manager.record_frame_timing(1, 100.0, 100.1)
        self._update(manager, frame_idx=1)
        manager.record_frame_timing(2, 101.0, 101.1)
        self._update(manager, frame_idx=2)
        lifecycle = manager.lifecycle_manager.get(1)
        self.assertIsNotNone(lifecycle)

        manager.flush_inactive(set(), frame_idx=3, capture_ts=102.0)
        self.assertIn(1, manager.tracks)
        self.assertFalse(lifecycle.closed)

        manager.flush_inactive(set(), frame_idx=4, capture_ts=105.9)
        self.assertIn(1, manager.tracks)
        self.assertFalse(lifecycle.closed)

        manager.flush_inactive(set(), frame_idx=5, capture_ts=106.1)
        self.assertTrue(lifecycle.closed)
        self.assertIn(5, manager.tracks[1]['events'])
        self.assertIn('TRACK_LOST_IN_ZONE_A_TIMEOUT', manager.tracks[1]['abnormal_reasons'])

    def test_verified_plate_handoff_reuses_lost_lifecycle_event_id(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        manager = EventManager(
            {
                'logic': {'plate_lock_frames': 3, 'event_trace_enabled': False},
                'event_capture_dir': temp_dir.name,
                'event_output_dir': temp_dir.name,
                'lane_name': 'lane-a',
            },
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_NoEventZoneManager(),
        )
        lifecycle = manager.lifecycle_manager.create(1, 'car', capture_ts=100.0)
        manager.lifecycle_manager.touch(
            1,
            capture_ts=100.0,
            plate_text='鲁A12345',
            plate_box=[10, 10, 30, 20],
            plate_edge='flow_start',
            plate_color='蓝色',
            plate_color_conf=0.92,
            plate_type='single',
        )
        manager.tracks[1] = {}
        manager.lifecycle_manager.mark_lost(1, capture_ts=101.0)

        for frame_idx in range(1, 4):
            manager.record_frame_timing(frame_idx, 102.0 + frame_idx * 0.1, 102.1 + frame_idx * 0.1)
            manager.update_track(
                track_id=2,
                plate_box=[11, 11, 31, 21],
                vehicle_box=[0, 0, 60, 60],
                plate_text='鲁A12345',
                frame_idx=frame_idx,
                frame=None,
                water_boxes=[],
                water_active=False,
                is_plate=True,
                vehicle_label='car',
                vehicle_conf=0.95,
                plate_conf=0.95,
                confirmed=True,
                anchor_point=(10.0, 10.0),
            )

        self.assertEqual(manager.tracks[2]['plate_text_locked'], '鲁A12345')
        self.assertIs(manager.lifecycle_manager.get(2), lifecycle)
        self.assertEqual(manager.tracks[2]['session_id'], lifecycle.event_id)
        self.assertEqual(lifecycle.active_tracker_id, 2)
        self.assertEqual(manager.tracks[2]['plate_color_locked'], '蓝色')
        self.assertAlmostEqual(manager.tracks[2]['plate_color_locked_conf'], 0.92)
        self.assertEqual(manager.tracks[2]['plate_color_locked_text'], '鲁A12345')

    def test_different_stable_plate_does_not_reuse_lost_lifecycle(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        manager = EventManager(
            {
                'logic': {'plate_lock_frames': 3, 'event_trace_enabled': False},
                'event_capture_dir': temp_dir.name,
                'event_output_dir': temp_dir.name,
                'lane_name': 'lane-a',
            },
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_NoEventZoneManager(),
        )
        lifecycle = manager.lifecycle_manager.create(1, 'car', capture_ts=100.0)
        manager.lifecycle_manager.touch(
            1,
            capture_ts=100.0,
            plate_text='鲁A12345',
            plate_box=[10, 10, 30, 20],
            plate_edge='flow_start',
        )
        manager.lifecycle_manager.mark_lost(1, capture_ts=101.0)

        for frame_idx in range(1, 4):
            manager.record_frame_timing(frame_idx, 102.0 + frame_idx * 0.1, 102.1 + frame_idx * 0.1)
            manager.update_track(
                track_id=2,
                plate_box=[11, 11, 31, 21],
                vehicle_box=[0, 0, 60, 60],
                plate_text='鲁A54321',
                frame_idx=frame_idx,
                frame=None,
                water_boxes=[],
                water_active=False,
                is_plate=True,
                vehicle_label='car',
                vehicle_conf=0.95,
                plate_conf=0.95,
                confirmed=True,
                anchor_point=(10.0, 10.0),
            )

        self.assertEqual(manager.tracks[2]['plate_text_locked'], '鲁A54321')
        self.assertIsNone(manager.lifecycle_manager.get(2))
        self.assertEqual(lifecycle.active_tracker_id, 0)

    def test_born_inside_track_defers_new_event_until_plate_attribution_resolves(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        uploader = _CollectingUploader()
        manager = EventManager(
            {
                'logic': {
                    'plate_lock_frames': 3,
                    'event_trace_enabled': False,
                    'min_track_frames_for_type1': 0,
                },
                'event_capture_dir': temp_dir.name,
                'event_output_dir': temp_dir.name,
                'lane_name': 'lane-a',
            },
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_BornInsideZoneManager(),
            uploader=uploader,
        )
        old_lifecycle = manager.lifecycle_manager.create(1, 'car', capture_ts=100.0)
        manager.lifecycle_manager.touch(
            1,
            capture_ts=100.0,
            plate_text='鲁A12345',
            plate_box=[10, 10, 30, 20],
            plate_edge='flow_start',
        )
        old_lifecycle.stages.update({1, 2})
        manager.lifecycle_manager.mark_lost(1, capture_ts=101.0)

        for frame_idx in range(1, 3):
            manager.record_frame_timing(frame_idx, 102.0 + frame_idx * 0.1, 102.1 + frame_idx * 0.1)
            manager.update_track(
                track_id=2,
                plate_box=None,
                vehicle_box=[0, 0, 60, 60],
                plate_text='',
                frame_idx=frame_idx,
                frame=None,
                water_boxes=[],
                water_active=False,
                is_plate=False,
                vehicle_label='car',
                vehicle_conf=0.95,
                plate_conf=None,
                confirmed=True,
                anchor_point=(10.0, 10.0),
            )

        self.assertTrue(manager.tracks[2]['lifecycle_handoff_pending'])
        self.assertTrue(manager.tracks[2]['deferred_type2'])
        self.assertEqual(manager.tracks[2]['events'], set())

        for frame_idx in range(3, 6):
            manager.record_frame_timing(frame_idx, 102.0 + frame_idx * 0.1, 102.1 + frame_idx * 0.1)
            manager.update_track(
                track_id=2,
                plate_box=[11, 11, 31, 21],
                vehicle_box=[0, 0, 60, 60],
                plate_text='鲁A54321',
                frame_idx=frame_idx,
                frame=None,
                water_boxes=[],
                water_active=False,
                is_plate=True,
                vehicle_label='car',
                vehicle_conf=0.95,
                plate_conf=0.95,
                confirmed=True,
                anchor_point=(10.0, 10.0),
            )

        new_lifecycle = manager.lifecycle_manager.get(2)
        self.assertIsNotNone(new_lifecycle)
        self.assertIsNot(new_lifecycle, old_lifecycle)
        self.assertEqual(manager.tracks[2]['events'], {1, 2})
        self.assertEqual([payload['type'] for payload in uploader.payloads], [1, 2])

    def test_lifecycle_plate_and_color_follow_confirmed_correction_after_type2(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        manager = EventManager(
            {
                'logic': {
                    'plate_lock_frames': 3,
                    'plate_correction_confirm_hits': 12,
                    'event_trace_enabled': False,
                },
                'event_capture_dir': temp_dir.name,
                'event_output_dir': temp_dir.name,
                'lane_name': 'lane-a',
            },
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_NoEventZoneManager(),
        )

        def update(frame_idx, text, color):
            manager.record_frame_timing(frame_idx, 100.0 + frame_idx * 0.1, 100.1 + frame_idx * 0.1)
            manager.update_track(
                track_id=1,
                plate_box=[10, 10, 30, 20],
                vehicle_box=[0, 0, 60, 60],
                plate_text=text,
                frame_idx=frame_idx,
                frame=None,
                water_boxes=[],
                water_active=False,
                is_plate=True,
                vehicle_label='car',
                vehicle_conf=0.95,
                plate_conf=0.95,
                confirmed=True,
                anchor_point=(10.0, 10.0),
                plate_color=color,
                plate_color_conf=0.95,
                plate_type='single',
            )

        for frame_idx in range(1, 4):
            update(frame_idx, '鲁A12345', '黄色')
        track_state = manager.tracks[1]
        manager._emit_event_core(
            track_id=1,
            event_type=2,
            frame_idx=3,
            frame=None,
            payload={'captureTime': '2026-07-18 10:00:01'},
            track_state=track_state,
            vehicle_type='car',
        )
        lifecycle = manager.lifecycle_manager.get(1)
        self.assertEqual(lifecycle.last_plate_text, '鲁A12345')
        self.assertEqual(lifecycle.last_plate_color, '黄色')

        for frame_idx in range(4, 16):
            update(frame_idx, '鲁A54321', '蓝色')

        self.assertEqual(track_state['plate_text_locked'], '鲁A54321')
        self.assertEqual(track_state['plate_color_locked'], '蓝色')
        self.assertEqual(lifecycle.last_plate_text, '鲁A54321')
        self.assertEqual(lifecycle.last_plate_color, '蓝色')

    def test_event_trace_writes_files_and_sanitizes_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'configs' / 'config.json'
            config_path.parent.mkdir(parents=True)
            config = {
                'config_path': str(config_path),
                'config_name': 'config.json',
                'system': {'device_id': 'device-a'},
                'video': {
                    'source': 'rtsp://user:secret@192.168.1.10:8557/live',
                    'source_mode': 'camera',
                },
                'zones': {},
                'lane_name': 'test',
                'logic': {
                    'event_trace_enabled': True,
                    'event_trace_dir': 'event_traces',
                    'event_trace_queue_size': 128,
                },
            }
            recorder = EventTraceRecorder.from_config(config, fps=25.0)
            run_dir = recorder.run_dir
            recorder.record('frame_tracks', {'frameIdx': 1, 'retainedTrackIds': [1]})
            recorder.record('event', {'event': {'type': 1, 'id': 'demo'}})
            recorder.close()

            self.assertTrue((run_dir / 'run.json').exists())
            self.assertTrue((run_dir / 'frames.jsonl').exists())
            self.assertTrue((run_dir / 'events.jsonl').exists())
            self.assertTrue((run_dir / 'summary.json').exists())
            metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
            self.assertNotIn('secret', metadata['source'])
            summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['eventTypeCounts'], {'1': 1})

    def _manager(self, uploader=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config = {
            'logic': {
                'plate_lock_frames': 3,
                'event_trace_enabled': False,
            },
            'event_capture_dir': temp_dir.name,
            'event_output_dir': temp_dir.name,
            'lane_name': 'lane-a',
            'track_timeout_frames': 225,
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_ScriptedZoneManager(),
            uploader=uploader,
        )

    @staticmethod
    def _update(manager, frame_idx):
        manager.update_track(
            track_id=1,
            plate_box=None,
            vehicle_box=[0, 0, 40, 40],
            plate_text='',
            frame_idx=frame_idx,
            frame=None,
            water_boxes=[],
            water_active=False,
            is_plate=False,
            vehicle_label='car',
            vehicle_conf=0.95,
            plate_conf=None,
            confirmed=False,
            anchor_point=(20.0, 40.0),
        )


if __name__ == '__main__':
    unittest.main()
