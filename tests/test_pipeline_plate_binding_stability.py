import unittest
import sys
import types


if "rknnlite" not in sys.modules:
    rknn_module = types.ModuleType("rknnlite")
    rknn_api_module = types.ModuleType("rknnlite.api")

    class _RKNNLiteStub:
        pass

    rknn_api_module.RKNNLite = _RKNNLiteStub
    rknn_module.api = rknn_api_module
    sys.modules["rknnlite"] = rknn_module
    sys.modules["rknnlite.api"] = rknn_api_module

from cleaningcar.pipeline import (
    _append_pending_plate_candidate,
    _cleanup_plate_binding_states,
    _cleanup_pending_plate_cache,
    _consume_pending_plate_candidates,
    _ensure_plate_binding_state,
    _refresh_car_plate_cache_from_locked,
    _stabilize_plate_binding,
)


class PlateBindingStabilityTests(unittest.TestCase):
    def test_single_frame_misbind_does_not_immediately_switch_locked_car(self):
        state = _ensure_plate_binding_state(frame_idx=1)

        _stabilize_plate_binding(
            state,
            frame_idx=1,
            candidate_car_id=101,
            candidate_score=0.70,
            locked_score=None,
        )
        _stabilize_plate_binding(
            state,
            frame_idx=2,
            candidate_car_id=101,
            candidate_score=0.72,
            locked_score=None,
        )
        self.assertEqual(state["locked_car_id"], 101)

        _stabilize_plate_binding(
            state,
            frame_idx=3,
            candidate_car_id=202,
            candidate_score=0.95,
            locked_score=0.40,
        )
        self.assertEqual(state["locked_car_id"], 101)
        self.assertEqual(state["candidate_car_id"], 202)
        self.assertEqual(state["candidate_hits"], 1)

        _stabilize_plate_binding(
            state,
            frame_idx=4,
            candidate_car_id=101,
            candidate_score=0.73,
            locked_score=0.73,
        )
        self.assertEqual(state["locked_car_id"], 101)
        self.assertIsNone(state["candidate_car_id"])
        self.assertEqual(state["candidate_hits"], 0)

    def test_switch_requires_three_consecutive_better_hits(self):
        state = _ensure_plate_binding_state(frame_idx=1)

        _stabilize_plate_binding(
            state,
            frame_idx=1,
            candidate_car_id=101,
            candidate_score=0.70,
            locked_score=None,
        )
        _stabilize_plate_binding(
            state,
            frame_idx=2,
            candidate_car_id=101,
            candidate_score=0.71,
            locked_score=None,
        )
        self.assertEqual(state["locked_car_id"], 101)

        _stabilize_plate_binding(
            state,
            frame_idx=3,
            candidate_car_id=202,
            candidate_score=0.91,
            locked_score=0.40,
        )
        _stabilize_plate_binding(
            state,
            frame_idx=4,
            candidate_car_id=202,
            candidate_score=0.92,
            locked_score=0.38,
        )
        self.assertEqual(state["locked_car_id"], 101)
        self.assertEqual(state["candidate_hits"], 2)

        _stabilize_plate_binding(
            state,
            frame_idx=5,
            candidate_car_id=202,
            candidate_score=0.93,
            locked_score=0.35,
        )
        self.assertEqual(state["locked_car_id"], 202)
        self.assertEqual(state["candidate_hits"], 0)
        self.assertIsNone(state["candidate_car_id"])

    def test_cache_only_refreshes_from_locked_bindings(self):
        state = _ensure_plate_binding_state(frame_idx=1)
        _stabilize_plate_binding(
            state,
            frame_idx=1,
            candidate_car_id=101,
            candidate_score=0.70,
            locked_score=None,
        )
        _stabilize_plate_binding(
            state,
            frame_idx=2,
            candidate_car_id=101,
            candidate_score=0.74,
            locked_score=None,
        )
        plate_states = {11: state}
        car_plate_cache = {}

        _refresh_car_plate_cache_from_locked(
            plate_binding_states=plate_states,
            car_plate_cache=car_plate_cache,
            active_car_ids={101},
            car_plate_cache_ttl=3,
        )
        self.assertEqual(car_plate_cache.get(101, {}).get("plate_id"), 11)
        car_to_plate = _refresh_car_plate_cache_from_locked(
            plate_binding_states=plate_states,
            car_plate_cache=car_plate_cache,
            active_car_ids={101},
            car_plate_cache_ttl=3,
        )
        plate_to_car = {plate_id: car_id for car_id, plate_id in car_to_plate.items()}
        self.assertEqual(plate_to_car.get(11), 101)

        _stabilize_plate_binding(
            state,
            frame_idx=3,
            candidate_car_id=202,
            candidate_score=0.95,
            locked_score=0.45,
        )
        _refresh_car_plate_cache_from_locked(
            plate_binding_states=plate_states,
            car_plate_cache=car_plate_cache,
            active_car_ids={101, 202},
            car_plate_cache_ttl=3,
        )
        self.assertNotIn(202, car_plate_cache)
        self.assertEqual(car_plate_cache.get(101, {}).get("plate_id"), 11)

    def test_cleanup_removes_stale_plate_binding_and_returns_locked_car(self):
        state = _ensure_plate_binding_state(frame_idx=1)
        _stabilize_plate_binding(
            state,
            frame_idx=1,
            candidate_car_id=101,
            candidate_score=0.70,
            locked_score=None,
        )
        _stabilize_plate_binding(
            state,
            frame_idx=2,
            candidate_car_id=101,
            candidate_score=0.72,
            locked_score=None,
        )
        plate_states = {11: state}

        removed_locked_ids = _cleanup_plate_binding_states(
            plate_binding_states=plate_states,
            frame_idx=30,
            active_car_ids=set(),
            plate_timeout_frames=5,
            vehicle_missing_frames=3,
        )

        self.assertNotIn(11, plate_states)
        self.assertIn(101, removed_locked_ids)

    def test_pending_plate_candidates_expire_before_late_vehicle_binding(self):
        pending = {}
        self.assertTrue(
            _append_pending_plate_candidate(
                pending,
                plate_id=11,
                text="鲁A12345",
                frame_idx=10,
                conf=0.91,
                box=[1, 2, 20, 12],
                max_entries=3,
            )
        )

        self.assertEqual(len(_consume_pending_plate_candidates(pending, 11, frame_idx=12, ttl_frames=5)), 1)

        self.assertTrue(
            _append_pending_plate_candidate(
                pending,
                plate_id=11,
                text="鲁A12345",
                frame_idx=10,
                conf=0.91,
                box=[1, 2, 20, 12],
                max_entries=3,
            )
        )
        self.assertEqual(_consume_pending_plate_candidates(pending, 11, frame_idx=40, ttl_frames=5), [])
        _cleanup_pending_plate_cache(pending, frame_idx=40, ttl_frames=5)
        self.assertNotIn(11, pending)


if __name__ == "__main__":
    unittest.main()
