from __future__ import annotations

import threading
import time
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np


RTP_BASE_FIELDS = ("seqnum-base", "clock-base")
VIDEO_ENCODINGS = {"H264", "H265", "HEVC"}


def safe_rtsp_source_label(source):
    text = str(source or "")
    try:
        parsed = urlsplit(text)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
            return text
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except Exception:
        return "rtsp://<redacted>" if text.lower().startswith(("rtsp://", "rtsps://")) else text


def redact_rtsp_credentials(message, source):
    result = str(message or "")
    raw_source = str(source or "")
    safe_source = safe_rtsp_source_label(raw_source)
    if raw_source:
        result = result.replace(raw_source, safe_source)
    try:
        parsed = urlsplit(raw_source)
        for secret in (parsed.username, parsed.password):
            if secret:
                result = result.replace(secret, "<redacted>")
    except Exception:
        pass
    return result


def is_target_video_rtp_caps(fields):
    """Return whether RTP CAPS belong to the wheel video stream."""
    values = fields or {}
    media = str(values.get("media") or "").strip().lower()
    encoding = str(values.get("encoding-name") or "").strip().upper()
    return media == "video" and encoding in VIDEO_ENCODINGS


def sanitize_wheel_rtp_caps(fields, enabled=True):
    """Remove only the broken RTSP RTP-Info bases from target video CAPS."""
    cleaned = deepcopy(dict(fields or {}))
    if not enabled or not is_target_video_rtp_caps(cleaned):
        return cleaned
    for field in RTP_BASE_FIELDS:
        cleaned.pop(field, None)
    return cleaned


def rtp_sequence_is_before(seqnum, seqnum_base):
    """Model RFC3550 16-bit ordering used by the deterministic regression fixture."""
    distance = (int(seqnum) - int(seqnum_base)) & 0xFFFF
    return distance >= 0x8000


def admitted_rtp_sequences(sequence, seqnum_base=None):
    """Small protocol fixture: expose which packets survive the advertised base gate."""
    admitted = []
    for value in sequence:
        seqnum = int(value) & 0xFFFF
        if seqnum_base is not None and rtp_sequence_is_before(seqnum, seqnum_base):
            continue
        admitted.append(seqnum)
    return admitted


def _load_gstreamer():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtsp", "1.0")
        from gi.repository import Gst, GstRtsp
    except Exception as exc:
        raise RuntimeError(f"PyGObject/GStreamer unavailable: {exc}") from exc
    Gst.init(None)
    return Gst, GstRtsp


def _gst_caps_to_dict(caps):
    values = {}
    if caps is None or caps.get_size() <= 0:
        return values
    structure = caps.get_structure(0)
    for index in range(structure.n_fields()):
        name = structure.nth_field_name(index)
        values[name] = structure.get_value(name)
    return values


def _sanitized_gst_caps(Gst, caps, enabled=True):
    if caps is None:
        return caps, False, {}
    fields = _gst_caps_to_dict(caps)
    if not enabled or not is_target_video_rtp_caps(fields):
        return caps, False, fields
    cleaned = Gst.Caps.new_empty()
    changed = False
    for index in range(caps.get_size()):
        structure = caps.get_structure(index).copy()
        structure_fields = {
            structure.nth_field_name(field_index): structure.get_value(structure.nth_field_name(field_index))
            for field_index in range(structure.n_fields())
        }
        if is_target_video_rtp_caps(structure_fields):
            for field in RTP_BASE_FIELDS:
                if structure.has_field(field):
                    structure.remove_field(field)
                    changed = True
        cleaned.append_structure(structure)
    return cleaned, changed, fields


def _parse_rtp_seqnum(buffer, map_flags):
    if buffer is None:
        return None
    ok, mapped = buffer.map(map_flags)
    if not ok:
        return None
    try:
        raw = mapped.data
        if raw is None or len(raw) < 4:
            return None
        return (int(raw[2]) << 8) | int(raw[3])
    finally:
        buffer.unmap(mapped)


class WheelGstCapture:
    """Wheel-only RTSP capture that ignores broken RTP-Info sequence bases."""

    def __init__(
        self,
        source,
        side,
        latency_ms=200,
        max_buffers=1,
        open_timeout_seconds=5.0,
        read_timeout_seconds=5.0,
        ignore_broken_rtp_info=True,
        reconnect_count=0,
        state_callback=None,
        cancel_event=None,
        gst_modules=None,
    ):
        self.source = str(source)
        self.side = str(side)
        self.latency_ms = max(0, int(latency_ms))
        self.max_buffers = max(1, int(max_buffers))
        self.open_timeout_seconds = max(0.1, float(open_timeout_seconds))
        self.read_timeout_seconds = max(0.0, float(read_timeout_seconds))
        self.ignore_broken_rtp_info = bool(ignore_broken_rtp_info)
        self.reconnect_count = max(0, int(reconnect_count))
        self._state_callback = state_callback
        self._cancel_event = cancel_event

        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._state = "starting"
        self._started_monotonic = time.monotonic()
        self._rtsp_ready_monotonic = 0.0
        self._first_rtp_monotonic = 0.0
        self._first_bgr_monotonic = 0.0
        self._first_rtp_seqnum = None
        self._advertised_seqnum_base = None
        self._advertised_clock_base = None
        self._last_error = ""
        self._caps_sanitized = False
        self._opened = False
        self._released = False
        self._frame_seq = 0
        self._sample_count = 0
        self._read_seq = 0
        self._latest_frame = None
        self._frame_cond = threading.Condition()
        self._first_frame_event = threading.Event()
        self._bus_stop = threading.Event()
        self._bus_thread = None
        self._pipeline = None
        self._source = None
        self._depay = None
        self._appsink = None
        self._probe_guards = set()
        self._probe_ids = []

        try:
            self._publish_state("starting")
            self.Gst, self.GstRtsp = gst_modules or _load_gstreamer()
            self._build_pipeline()
            self._start_pipeline()
        except Exception as exc:
            self._fail(f"pipeline_init_failed: {exc}")
            self.release()

    def _make(self, factory, name):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        return element

    def _build_pipeline(self):
        Gst = self.Gst
        self._pipeline = Gst.Pipeline.new(f"wheel-{self.side}")
        if self._pipeline is None:
            raise RuntimeError("failed to create GStreamer pipeline")

        self._source = self._make("rtspsrc", "wheel-source")
        self._depay = self._make("rtph265depay", "wheel-depay")
        parser = self._make("h265parse", "wheel-parser")
        decoder = self._make("mppvideodec", "wheel-decoder")
        capsfilter = self._make("capsfilter", "wheel-bgr-caps")
        self._appsink = self._make("appsink", "wheel-appsink")

        self._source.set_property("location", self.source)
        self._source.set_property("latency", self.latency_ms)
        self._source.set_property("protocols", self.GstRtsp.RTSPLowerTrans.TCP)
        parser.set_property("config-interval", -1)
        decoder.set_property("format", "BGR")
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)
        self._appsink.set_property("drop", True)
        self._appsink.set_property("max-buffers", self.max_buffers)

        for element in (self._source, self._depay, parser, decoder, capsfilter, self._appsink):
            self._pipeline.add(element)
        if not self._depay.link(parser) or not parser.link(decoder) or not decoder.link(capsfilter) or not capsfilter.link(self._appsink):
            raise RuntimeError("failed to link wheel direct-BGR pipeline")

        self._source.connect("new-manager", self._on_new_manager)
        self._source.connect("pad-added", self._on_rtsp_pad_added)
        self._appsink.connect("new-sample", self._on_new_sample)

    def _start_pipeline(self):
        Gst = self.Gst
        result = self._pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline refused PLAYING state")
        self._bus_thread = threading.Thread(target=self._bus_loop, name=f"wheel-gst-bus-{self.side}", daemon=True)
        self._bus_thread.start()
        deadline = time.monotonic() + self.open_timeout_seconds
        while not self._first_frame_event.wait(timeout=min(0.1, max(0.0, deadline - time.monotonic()))):
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise RuntimeError("open_cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"first BGR frame timeout after {self.open_timeout_seconds:.2f}s")
        self._opened = self._latest_frame is not None and self._state == "bgr_ready"

    def _on_new_manager(self, _source, manager):
        manager.connect("new-jitterbuffer", self._on_new_jitterbuffer)

    def _on_new_jitterbuffer(self, _manager, jitterbuffer, *_args):
        sink_pad = jitterbuffer.get_static_pad("sink")
        if sink_pad is None:
            return
        mask = self.Gst.PadProbeType.EVENT_DOWNSTREAM | self.Gst.PadProbeType.BUFFER
        probe_id = sink_pad.add_probe(mask, self._on_jitterbuffer_sink_probe)
        self._probe_ids.append((sink_pad, probe_id))

    def _on_jitterbuffer_sink_probe(self, pad, info):
        Gst = self.Gst
        info_type = info.type
        event = info.get_event() if info_type & Gst.PadProbeType.EVENT_DOWNSTREAM else None
        if event is not None and event.type == Gst.EventType.CAPS:
            caps = event.parse_caps()
            cleaned, changed, original = _sanitized_gst_caps(
                Gst,
                caps,
                enabled=self.ignore_broken_rtp_info,
            )
            if is_target_video_rtp_caps(original):
                if "seqnum-base" in original and self._advertised_seqnum_base is None:
                    self._advertised_seqnum_base = original.get("seqnum-base")
                if "clock-base" in original and self._advertised_clock_base is None:
                    self._advertised_clock_base = original.get("clock-base")
            if changed:
                guard = id(pad)
                if guard not in self._probe_guards:
                    self._probe_guards.add(guard)
                    try:
                        accepted = pad.send_event(Gst.Event.new_caps(cleaned))
                    finally:
                        self._probe_guards.discard(guard)
                    if accepted:
                        self._caps_sanitized = True
                        return Gst.PadProbeReturn.DROP

        buffer = info.get_buffer() if info_type & Gst.PadProbeType.BUFFER else None
        if buffer is not None and self._first_rtp_monotonic <= 0.0:
            current_caps = pad.get_current_caps()
            if is_target_video_rtp_caps(_gst_caps_to_dict(current_caps)):
                seqnum = _parse_rtp_seqnum(buffer, Gst.MapFlags.READ)
                if seqnum is not None:
                    self._first_rtp_seqnum = seqnum
                    self._first_rtp_monotonic = time.monotonic()
                    self._publish_state("rtp_ready")
        return Gst.PadProbeReturn.OK

    def _on_rtsp_pad_added(self, _source, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        fields = _gst_caps_to_dict(caps)
        if not is_target_video_rtp_caps(fields):
            return
        sink_pad = self._depay.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result == self.Gst.PadLinkReturn.OK:
            self._rtsp_ready_monotonic = time.monotonic()

    def _sample_to_frame(self, sample):
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        fps = 0.0
        try:
            ok, numerator, denominator = structure.get_fraction("framerate")
            if ok and denominator:
                fps = float(numerator) / float(denominator)
        except Exception:
            fps = 0.0
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map BGR sample")
        try:
            raw = np.frombuffer(mapped.data, dtype=np.uint8)
            row_bytes = width * 3
            if height <= 0 or row_bytes <= 0 or raw.size < height * row_bytes:
                raise RuntimeError(f"invalid BGR sample size={raw.size} width={width} height={height}")
            stride = raw.size // height
            if stride < row_bytes:
                raise RuntimeError(f"invalid BGR stride={stride} row_bytes={row_bytes}")
            frame = raw[: height * stride].reshape(height, stride)[:, :row_bytes].reshape(height, width, 3).copy()
        finally:
            buffer.unmap(mapped)
        return frame, width, height, fps

    def _on_new_sample(self, appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return self.Gst.FlowReturn.ERROR
            frame, width, height, fps = self._sample_to_frame(sample)
            now = time.monotonic()
            with self._frame_cond:
                self.width = width
                self.height = height
                if fps > 0.0:
                    self.fps = fps
                self._frame_seq += 1
                self._sample_count += 1
                self._latest_frame = frame
                if self._first_bgr_monotonic <= 0.0:
                    self._first_bgr_monotonic = now
                    self._publish_state("bgr_ready")
                    self._first_frame_event.set()
                self._frame_cond.notify_all()
            return self.Gst.FlowReturn.OK
        except Exception as exc:
            self._fail(f"sample_error: {exc}")
            return self.Gst.FlowReturn.ERROR

    def _bus_loop(self):
        Gst = self.Gst
        bus = self._pipeline.get_bus() if self._pipeline is not None else None
        if bus is None:
            return
        mask = Gst.MessageType.ERROR | Gst.MessageType.EOS
        while not self._bus_stop.is_set():
            message = bus.timed_pop_filtered(int(0.2 * Gst.SECOND), mask)
            if message is None:
                continue
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self._fail(f"gstreamer_error: {error}; debug={debug or ''}")
            else:
                self._fail("gstreamer_eos")
            break

    def _fail(self, message):
        self._last_error = redact_rtsp_credentials(message, self.source)
        self._publish_state("failed")
        self._opened = False
        self._first_frame_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()

    def _publish_state(self, state):
        self._state = str(state)
        callback = self._state_callback
        if callback is None:
            return
        try:
            callback(self._state)
        except Exception:
            pass

    def isOpened(self):
        return bool(self._opened and not self._released and self._state == "bgr_ready")

    def read(self):
        if not self.isOpened():
            return False, None
        deadline = None if self.read_timeout_seconds <= 0.0 else time.monotonic() + self.read_timeout_seconds
        with self._frame_cond:
            while self._frame_seq <= self._read_seq and self.isOpened():
                if deadline is None:
                    self._frame_cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._fail(f"read_timeout after {self.read_timeout_seconds:.2f}s")
                    return False, None
                self._frame_cond.wait(timeout=remaining)
            if not self.isOpened() or self._latest_frame is None:
                return False, None
            self._read_seq = self._frame_seq
            return True, self._latest_frame.copy()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def diagnostics(self):
        started = self._started_monotonic
        return {
            "side": self.side,
            "state": self._state,
            "rtsp_negotiation_seconds": (
                self._rtsp_ready_monotonic - started if self._rtsp_ready_monotonic > 0.0 else -1.0
            ),
            "first_rtp_seconds": (
                self._first_rtp_monotonic - started if self._first_rtp_monotonic > 0.0 else -1.0
            ),
            "first_bgr_seconds": (
                self._first_bgr_monotonic - started if self._first_bgr_monotonic > 0.0 else -1.0
            ),
            "first_rtp_seqnum": self._first_rtp_seqnum,
            "advertised_seqnum_base": self._advertised_seqnum_base,
            "advertised_clock_base": self._advertised_clock_base,
            "ignore_broken_rtp_info": self.ignore_broken_rtp_info,
            "caps_sanitized": self._caps_sanitized,
            "reconnect_count": self.reconnect_count,
            "last_read_error": self._last_error,
            "recent_error_match_count": int(bool(self._last_error)),
            "recent_error_lines": [self._last_error] if self._last_error else [],
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sample_count": self._sample_count,
        }

    def release(self):
        if self._released:
            return
        self._released = True
        self._opened = False
        self._bus_stop.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.set_state(self.Gst.State.NULL)
                pipeline.get_state(int(2 * self.Gst.SECOND))
            except Exception:
                pass
        for pad, probe_id in self._probe_ids:
            try:
                if probe_id:
                    pad.remove_probe(probe_id)
            except Exception:
                pass
        self._probe_ids.clear()
        bus_thread = self._bus_thread
        if bus_thread is not None and bus_thread.is_alive() and threading.current_thread() is not bus_thread:
            bus_thread.join(timeout=1.0)
        if self._state != "failed":
            self._publish_state("released")


def create_wheel_gstreamer_capture(
    source,
    side,
    latency_ms=200,
    max_buffers=1,
    open_timeout_seconds=5.0,
    read_timeout_seconds=5.0,
    ignore_broken_rtp_info=True,
    reconnect_count=0,
    state_callback=None,
    cancel_event=None,
):
    return WheelGstCapture(
        source=source,
        side=side,
        latency_ms=latency_ms,
        max_buffers=max_buffers,
        open_timeout_seconds=open_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        ignore_broken_rtp_info=ignore_broken_rtp_info,
        reconnect_count=reconnect_count,
        state_callback=state_callback,
        cancel_event=cancel_event,
    )
