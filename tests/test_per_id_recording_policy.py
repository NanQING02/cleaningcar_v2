import sys
import types
import unittest


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
    _resolve_per_id_recording_params,
    _resolve_per_id_video_source,
    _resolve_raw_per_id_prebuffer_frames,
    _should_drop_stale_frames,
)


class PerIdRecordingPolicyTests(unittest.TestCase):
    def test_file_source_keeps_all_frames(self):
        self.assertFalse(_should_drop_stale_frames("file"))

    def test_camera_source_keeps_realtime_drop_behavior(self):
        self.assertTrue(_should_drop_stale_frames("camera"))
        self.assertTrue(_should_drop_stale_frames("auto"))

    def test_per_id_recording_uses_source_size_source_fps_and_every_frame(self):
        params = _resolve_per_id_recording_params(
            width=704,
            height=576,
            source_fps=18.0,
            logic_cfg={
                "per_id_downscale_ratio": 0.5,
                "per_id_target_width": 1280,
                "per_id_target_height": 720,
                "per_id_fps": 20.0,
                "per_id_frame_stride": 2,
            },
        )

        self.assertEqual(
            params,
            {
                "width": 704,
                "height": 576,
                "fps": 18.0,
                "frame_stride": 1,
            },
        )

    def test_per_id_video_auto_uses_raw_when_drawing_is_disabled(self):
        self.assertEqual(_resolve_per_id_video_source({}, no_draw=True), "raw")
        self.assertEqual(_resolve_per_id_video_source({}, no_draw=False), "annotated")

    def test_per_id_video_source_can_be_forced(self):
        self.assertEqual(_resolve_per_id_video_source({"per_id_video_source": "raw"}, no_draw=False), "raw")
        self.assertEqual(_resolve_per_id_video_source({"per_id_video_source": "annotated"}, no_draw=True), "annotated")

    def test_raw_per_id_prebuffer_uses_source_fps(self):
        self.assertEqual(
            _resolve_raw_per_id_prebuffer_frames(
                25.0,
                {"per_id_raw_prebuffer_seconds": 3.0},
            ),
            75,
        )
        self.assertEqual(
            _resolve_raw_per_id_prebuffer_frames(
                25.0,
                {"per_id_raw_prebuffer_seconds": 0.0},
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
