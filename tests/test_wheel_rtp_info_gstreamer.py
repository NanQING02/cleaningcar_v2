import struct
import unittest

import numpy as np

from cleaningcar.wheel_gstreamer import WheelGstCapture, _sanitized_gst_caps


try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    GST_AVAILABLE = Gst.ElementFactory.find("rtpjitterbuffer") is not None
except Exception:
    Gst = None
    GST_AVAILABLE = False


@unittest.skipUnless(GST_AVAILABLE, "需要带 rtpjitterbuffer 的 PyGObject/GStreamer")
class GStreamerRtpInfoRegressionTests(unittest.TestCase):
    def _caps(self, seqnum_base=None):
        fields = [
            "application/x-rtp",
            "media=(string)video",
            "payload=(int)96",
            "clock-rate=(int)90000",
            "encoding-name=(string)H265",
            "clock-base=(uint)0",
            "ssrc=(uint)1234",
            "sprop-vps=(string)vps",
            "sprop-sps=(string)sps",
            "sprop-pps=(string)pps",
        ]
        if seqnum_base is not None:
            fields.append(f"seqnum-base=(uint){int(seqnum_base)}")
        return Gst.Caps.from_string(",".join(fields))

    @staticmethod
    def _packet(seqnum, timestamp):
        raw = struct.pack("!BBHII", 0x80, 0x80 | 96, seqnum & 0xFFFF, timestamp & 0xFFFFFFFF, 1234)
        raw += b"payload"
        buffer = Gst.Buffer.new_allocate(None, len(raw), None)
        buffer.fill(0, raw)
        return buffer

    def _run_harness(self, caps, sequence, timeout_ms=300):
        pipeline = Gst.parse_launch(
            "appsrc name=src is-live=true format=time do-timestamp=true "
            "! rtpjitterbuffer latency=0 "
            "! appsink name=sink sync=false emit-signals=false"
        )
        source = pipeline.get_by_name("src")
        sink = pipeline.get_by_name("sink")
        source.set_property("caps", caps)
        output = []
        try:
            self.assertNotEqual(pipeline.set_state(Gst.State.PLAYING), Gst.StateChangeReturn.FAILURE)
            pipeline.get_state(Gst.SECOND)
            for index, seqnum in enumerate(sequence):
                result = source.emit("push-buffer", self._packet(seqnum, index * 3600))
                self.assertEqual(result, Gst.FlowReturn.OK)
            source.emit("end-of-stream")
            deadline = Gst.util_get_timestamp() + timeout_ms * Gst.MSECOND
            while Gst.util_get_timestamp() < deadline:
                sample = sink.emit("try-pull-sample", 20 * Gst.MSECOND)
                if sample is None:
                    continue
                buffer = sample.get_buffer()
                ok, mapped = buffer.map(Gst.MapFlags.READ)
                self.assertTrue(ok)
                try:
                    output.append((mapped.data[2] << 8) | mapped.data[3])
                finally:
                    buffer.unmap(mapped)
            return output
        finally:
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(Gst.SECOND)

    def test_broken_high_half_is_dropped_before_fix_and_immediate_after_fix(self):
        broken = self._caps(seqnum_base=1)
        cleaned, changed, original = _sanitized_gst_caps(Gst, broken, enabled=True)

        self.assertTrue(changed)
        self.assertEqual(original["seqnum-base"], 1)
        self.assertEqual(self._run_harness(broken, [52713]), [])
        self.assertEqual(self._run_harness(cleaned, [52713]), [52713])

    def test_compliant_and_low_sequence_inputs_remain_readable(self):
        compliant = self._caps(seqnum_base=5102)
        cleaned, changed, _ = _sanitized_gst_caps(Gst, compliant, enabled=True)

        self.assertTrue(changed)
        self.assertEqual(self._run_harness(compliant, [5102, 5103]), [5102, 5103])
        self.assertEqual(self._run_harness(cleaned, [5102, 5103]), [5102, 5103])

    def test_16_bit_wrap_is_continuous_after_fix(self):
        cleaned, changed, _ = _sanitized_gst_caps(Gst, self._caps(seqnum_base=1), enabled=True)
        sequence = [65534, 65535, 0, 1, 2]

        self.assertTrue(changed)
        self.assertEqual(self._run_harness(cleaned, sequence), sequence)

    def test_bgr_sample_preserves_dimensions_and_channel_order(self):
        expected = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        buffer = Gst.Buffer.new_allocate(None, expected.nbytes, None)
        buffer.fill(0, expected.tobytes())
        caps = Gst.Caps.from_string("video/x-raw,format=BGR,width=2,height=2,framerate=20/1")
        sample = Gst.Sample.new(buffer, caps, None, None)
        capture = object.__new__(WheelGstCapture)
        capture.Gst = Gst

        frame, width, height, fps = capture._sample_to_frame(sample)

        np.testing.assert_array_equal(frame, expected)
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(fps, 20.0)


if __name__ == "__main__":
    unittest.main()
