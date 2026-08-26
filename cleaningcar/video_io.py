import os
import select
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np

FFMPEG_HW_ENCODERS = ('h264_rkmpp', 'h264_v4l2m2m', 'h264_omx')
FFMPEG_HW_DECODER_CANDIDATES = (
    'h264_rkmpp',
    'hevc_rkmpp',
    'mjpeg_rkmpp',
    'mpeg2_rkmpp',
    'vp8_rkmpp',
    'vp9_rkmpp',
)
DELETE_UNQUALIFIED_PER_ID_VIDEO = True

FFMPEG_CODEC_TO_RKMPP_DECODER = {
    'h264': 'h264_rkmpp',
    'hevc': 'hevc_rkmpp',
    'h265': 'hevc_rkmpp',
    'mjpeg': 'mjpeg_rkmpp',
    'mpeg2video': 'mpeg2_rkmpp',
    'vp8': 'vp8_rkmpp',
    'vp9': 'vp9_rkmpp',
}


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value, default=False):
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    if value is None:
        return bool(default)
    return bool(value)


def _gstreamer_bgr_mode(value=None):
    raw = str(value or os.environ.get('CLEANINGCAR_GSTREAMER_BGR_MODE', '') or '').strip().lower()
    if raw in {'safe', 'safe_bgr', 'videoconvert'}:
        return 'safe'
    if raw in {'direct', 'direct_bgr'}:
        return 'direct'
    return 'direct'


def _parse_avg_frame_rate(text):
    raw = str(text or '').strip()
    if not raw or raw in ('0/0', 'N/A'):
        return 0.0
    if '/' in raw:
        left, right = raw.split('/', 1)
        try:
            num = float(left)
            den = float(right)
            if den != 0.0:
                return num / den
        except (TypeError, ValueError):
            return 0.0
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _probe_ffmpeg_stream(src):
    if not isinstance(src, str):
        return None
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=codec_name,width,height,avg_frame_rate',
        '-of',
        'default=noprint_wrappers=1:nokey=0',
    ]
    if src.startswith(('rtsp://', 'rtsps://')):
        cmd.extend(['-rtsp_transport', 'tcp'])
    cmd.append(src)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None

    info = {}
    for line in (proc.stdout or '').splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        info[key.strip()] = value.strip()
    try:
        width = int(info.get('width', '0') or 0)
        height = int(info.get('height', '0') or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {
        'codec_name': str(info.get('codec_name', '') or '').strip().lower(),
        'width': width,
        'height': height,
        'fps': _parse_avg_frame_rate(info.get('avg_frame_rate')),
    }


class FfmpegRawVideoCapture:
    STDERR_HISTORY_LIMIT = 80

    def __init__(
        self,
        src,
        decoder,
        stream_info,
        rtsp_latency_ms=200,
        read_timeout_seconds=5.0,
        output_width=0,
        output_height=0,
        video_filter='',
        use_drm_prime=False,
        backend='ffmpeg_rawvideo',
    ):
        self.src = str(src)
        self.decoder = str(decoder or '').strip()
        self.input_width = int((stream_info or {}).get('width') or 0)
        self.input_height = int((stream_info or {}).get('height') or 0)
        self.width = int(output_width or self.input_width)
        self.height = int(output_height or self.input_height)
        self.fps = float((stream_info or {}).get('fps') or 0.0)
        self.codec_name = str((stream_info or {}).get('codec_name') or '').strip().lower()
        self.frame_bytes = max(0, self.width * self.height * 3)
        self.read_timeout_seconds = max(0.0, float(read_timeout_seconds or 0.0))
        self.proc = None
        self.backend = str(backend or 'ffmpeg_rawvideo')
        self.video_filter = str(video_filter or '').strip()
        self.use_drm_prime = bool(use_drm_prime)
        self._opened = False
        self._stderr_thread = None
        self._stderr_stop = threading.Event()
        self._stderr_lines = deque(maxlen=self.STDERR_HISTORY_LIMIT)
        self._recent_error_lines = deque(maxlen=12)
        self._recent_error_match_count = 0
        self._last_read_error = ''
        self._last_read_error_ts = 0.0
        if self.width <= 0 or self.height <= 0 or self.frame_bytes <= 0 or not self.decoder:
            return
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'warning',
        ]
        if self.src.startswith(('rtsp://', 'rtsps://')):
            cmd.extend([
                '-rtsp_transport',
                'tcp',
                '-fflags',
                'nobuffer',
                '-flags',
                'low_delay',
                '-max_delay',
                str(max(0, int(rtsp_latency_ms)) * 1000),
            ])
        if self.use_drm_prime:
            cmd.extend([
                '-hwaccel',
                'rkmpp',
                '-hwaccel_output_format',
                'drm_prime',
            ])
        cmd.extend([
            '-c:v',
            self.decoder,
            '-i',
            self.src,
            '-an',
        ])
        if self.video_filter:
            cmd.extend(['-vf', self.video_filter])
        cmd.extend([
            '-pix_fmt',
            'bgr24',
            '-f',
            'rawvideo',
            'pipe:1',
        ])
        self.command = list(cmd)
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=max(self.frame_bytes * 4, 1024 * 1024),
            )
            self._opened = bool(self.proc.stdout is not None)
            if self._opened and self.proc.stderr is not None:
                self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
                self._stderr_thread.start()
        except Exception:
            self.proc = None
            self._opened = False

    def isOpened(self):
        return bool(
            self._opened
            and self.proc is not None
            and self.proc.poll() is None
            and self.proc.stdout is not None
        )

    @staticmethod
    def _is_error_stderr_line(line):
        text = str(line or '').strip().lower()
        if not text:
            return False
        markers = (
            'error',
            'failed',
            'invalid',
            'timeout',
            'pps',
            'slice_header',
            'mpp',
            'nal unit',
            'decode',
            'broken',
            'mismatch',
        )
        return any(marker in text for marker in markers)

    def _remember_read_error(self, message):
        self._last_read_error = str(message or '').strip()
        self._last_read_error_ts = time.time()

    def _drain_stderr(self):
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            while not self._stderr_stop.is_set():
                line = proc.stderr.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                text = line.decode('utf-8', errors='replace').strip()
                if not text:
                    continue
                self._stderr_lines.append(text)
                if self._is_error_stderr_line(text):
                    self._recent_error_lines.append(text)
                    self._recent_error_match_count += 1
        except Exception:
            return

    def diagnostics(self):
        return {
            'backend': self.backend,
            'decoder': self.decoder,
            'codec_name': self.codec_name,
            'input_width': getattr(self, 'input_width', self.width),
            'input_height': getattr(self, 'input_height', self.height),
            'output_width': self.width,
            'output_height': self.height,
            'video_filter': getattr(self, 'video_filter', ''),
            'read_timeout_seconds': self.read_timeout_seconds,
            'last_read_error': self._last_read_error,
            'last_read_error_ts': self._last_read_error_ts,
            'recent_stderr_lines': list(self._stderr_lines)[-8:],
            'recent_error_lines': list(self._recent_error_lines),
            'recent_error_match_count': int(self._recent_error_match_count),
        }

    def read(self):
        if not self.isOpened():
            return False, None
        try:
            if self.read_timeout_seconds <= 0.0:
                buf = self.proc.stdout.read(self.frame_bytes)
            else:
                buf = self._read_exact_with_timeout(self.frame_bytes, self.read_timeout_seconds)
        except TimeoutError as exc:
            self._remember_read_error(str(exc))
            self.release()
            return False, None
        except Exception as exc:
            self._remember_read_error(f'read_exception: {exc}')
            self.release()
            return False, None
        if len(buf) != self.frame_bytes:
            self._remember_read_error(f'short_read: got={len(buf)} expected={self.frame_bytes}')
            self.release()
            return False, None
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def _read_exact_with_timeout(self, size, timeout_seconds):
        if self.proc is None or self.proc.stdout is None:
            raise TimeoutError('capture_closed')
        fd = self.proc.stdout.fileno()
        chunks = bytearray()
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f'ffmpeg_rawvideo_read_timeout after {timeout_seconds:.2f}s '
                    f'bytes={len(chunks)}/{size} decoder={self.decoder}'
                )
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def release(self):
        proc = self.proc
        self.proc = None
        self._opened = False
        if proc is None:
            return
        self._stderr_stop.set()
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if (
            self._stderr_thread is not None
            and self._stderr_thread.is_alive()
            and threading.current_thread() is not self._stderr_thread
        ):
            self._stderr_thread.join(timeout=0.2)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0


class TimedVideoCapture:
    def __init__(self, cap, read_timeout_seconds=5.0, backend='opencv', source=''):
        self._cap = cap
        self.read_timeout_seconds = max(0.0, float(read_timeout_seconds or 0.0))
        self.backend = str(backend or 'opencv')
        self.source = str(source or '')
        self._opened = self._cap_is_opened()
        self._last_read_error = ''
        self._last_read_error_ts = 0.0
        self._recent_error_match_count = 0

    def _cap_is_opened(self):
        if self._cap is None or not hasattr(self._cap, 'isOpened'):
            return False
        try:
            return bool(self._cap.isOpened())
        except Exception:
            return False

    def isOpened(self):
        return bool(self._opened and self._cap_is_opened())

    def _remember_read_error(self, message):
        self._last_read_error = str(message or '').strip()
        self._last_read_error_ts = time.time()
        self._recent_error_match_count += 1

    def read(self):
        if not self.isOpened():
            return False, None
        if self.read_timeout_seconds <= 0.0:
            try:
                return self._cap.read()
            except Exception as exc:
                self._remember_read_error(f'timed_capture_read_exception: {exc}')
                self.release()
                return False, None

        result_q = Queue(maxsize=1)

        def _read_once():
            try:
                result_q.put(self._cap.read(), block=False)
            except Exception as exc:
                result_q.put(exc, block=False)

        thread = threading.Thread(target=_read_once, daemon=True)
        thread.start()
        try:
            result = result_q.get(timeout=self.read_timeout_seconds)
        except Empty:
            self._remember_read_error(
                f'timed_capture_read_timeout after {self.read_timeout_seconds:.2f}s backend={self.backend}'
            )
            self.release()
            return False, None

        if isinstance(result, Exception):
            self._remember_read_error(f'timed_capture_read_exception: {result}')
            self.release()
            return False, None
        try:
            ok, frame = result
        except Exception:
            self._remember_read_error(f'timed_capture_bad_read_result: {type(result).__name__}')
            self.release()
            return False, None
        return bool(ok), frame

    def release(self):
        cap = self._cap
        self._cap = None
        self._opened = False
        if cap is None or not hasattr(cap, 'release'):
            return
        try:
            cap.release()
        except Exception:
            pass

    def get(self, prop_id):
        if self._cap is None or not hasattr(self._cap, 'get'):
            return 0.0
        try:
            return self._cap.get(prop_id)
        except Exception:
            return 0.0

    def set(self, prop_id, value):
        if self._cap is None or not hasattr(self._cap, 'set'):
            return False
        try:
            return bool(self._cap.set(prop_id, value))
        except Exception:
            return False

    def diagnostics(self):
        return {
            'backend': self.backend,
            'source': self.source,
            'read_timeout_seconds': self.read_timeout_seconds,
            'last_read_error': self._last_read_error,
            'last_read_error_ts': self._last_read_error_ts,
            'recent_error_lines': [],
            'recent_error_match_count': int(self._recent_error_match_count),
        }


class FfmpegH264Writer:
    def __init__(self, path, width, height, fps, encoders=None):
        self.path = str(path)
        p = Path(self.path)
        if p.suffix:
            temp_name = p.stem + '_temp' + p.suffix
        else:
            temp_name = p.name + '_temp'
        self._output_path = str(p.with_name(temp_name))
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.proc = None
        self.stdin = None
        self.encoder = None
        self.backend = 'ffmpeg'
        self._encoders = tuple(encoders or FFMPEG_HW_ENCODERS)
        self._opened = False
        self._frames_total = 0
        self._frames_since_log = 0
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._log_interval = 10.0
        self._start()

    def _build_cmd(self, encoder):
        base = [
            'ffmpeg',
            '-y',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'bgr24',
            '-s',
            f'{self.width}x{self.height}',
            '-r',
            f'{self.fps}',
            '-i',
            '-',
            '-an',
        ]
        if encoder not in FFMPEG_HW_ENCODERS:
            raise ValueError(f'unsupported hardware encoder: {encoder}')
        opts = [
            '-c:v',
            encoder,
            '-pix_fmt',
            'yuv420p',
        ]
        tail = [
            '-movflags',
            '+faststart',
            self._output_path,
        ]
        return base + opts + tail

    def _try_start(self, encoder):
        self.proc = None
        self.stdin = None
        self.encoder = None
        self._opened = False
        cmd = self._build_cmd(encoder)
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.stdin = self.proc.stdin
            self.encoder = encoder
            # Wait only briefly — if FFmpeg process hasn't exited after 10ms
            # it means the encoder started successfully on this hardware.
            time.sleep(0.01)
            if self.proc.poll() is not None:
                self.proc = None
                self.stdin = None
                self.encoder = None
                self._opened = False
                print(f'[per-id-video] encoder {encoder} exited immediately for {self.path}, falling back')
                return False
            self._opened = True
            print(f'[per-id-video] using ffmpeg encoder={encoder} path={self.path}')
            return True
        except Exception as exc:
            self.proc = None
            self.stdin = None
            self.encoder = None
            self._opened = False
            print(f'[per-id-video] failed to start ffmpeg encoder {encoder} for {self.path}: {exc}')
            return False

    def _start(self):
        for enc in self._encoders:
            if self._try_start(enc):
                return
        print(f'[per-id-video] no available H.264 encoder for {self.path}')

    def is_opened(self):
        if not self._opened or not self.proc or not self.stdin:
            return False
        if self.proc.poll() is not None:
            return False
        return True

    def write(self, frame):
        if not self.is_opened():
            return
        if frame is None:
            return
        try:
            self.stdin.write(frame.tobytes())
            self._frames_total += 1
            self._frames_since_log += 1
            now = time.time()
            if self._log_interval > 0 and now - self._last_log_time >= self._log_interval:
                elapsed = now - self._last_log_time
                fps = self._frames_since_log / max(elapsed, 1e-6)
                print(f'[per-id-video] encoder={self.encoder} fps={fps:.2f} window={elapsed:.1f}s total_frames={self._frames_total} path={self.path}')
                self._frames_since_log = 0
                self._last_log_time = now
        except Exception as exc:
            print(f'[per-id-video] write failed for {self.path}: {exc}')
            self.release()

    def release(self):
        finalized = False
        if self.stdin:
            try:
                self.stdin.close()
            except Exception:
                pass
            self.stdin = None
        exit_code = None
        if self.proc:
            try:
                self.proc.wait(timeout=60.0)
                exit_code = self.proc.returncode
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        self._opened = False
        if getattr(self, '_output_path', None) and self.path:
            try:
                if os.path.exists(self._output_path):
                    if exit_code is None or exit_code != 0:
                        print(f'[per-id-video] ffmpeg exit code {exit_code} for {self._output_path}, not renaming')
                    else:
                        os.replace(self._output_path, self.path)
                        print(f'[per-id-video] finalized video: {self.path}')
                        finalized = True
            except Exception as exc:
                print(f'[per-id-video] rename failed {self._output_path} -> {self.path}: {exc}')
        return finalized


class AsyncPerIdVideoWriter:
    """Bounded async wrapper so video encoding never blocks the main result path."""

    def __init__(self, writer, resize_fn=None, queue_size=8, log_interval=10.0):
        self.writer = writer
        self.resize_fn = resize_fn
        self.queue_size = max(1, int(queue_size or 1))
        self._queue = Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='per-id-video-writer', daemon=True)
        self._released = False
        self._lock = threading.Lock()
        self._enqueued = 0
        self._written = 0
        self._dropped = 0
        self._write_errors = 0
        self._last_log_time = time.time()
        self._log_interval = float(log_interval or 0.0)
        self.path = getattr(writer, 'path', '')
        self.backend = getattr(writer, 'backend', '')
        self.encoder = getattr(writer, 'encoder', '')
        self._thread.start()

    def is_opened(self):
        return bool(self.writer is not None and self.writer.is_opened())

    def write(self, frame):
        if frame is None or self._released or not self.is_opened():
            return
        try:
            self._queue.put_nowait(frame)
            with self._lock:
                self._enqueued += 1
            return
        except Full:
            pass

        try:
            self._queue.get_nowait()
            self._queue.task_done()
            with self._lock:
                self._dropped += 1
        except Empty:
            pass

        try:
            self._queue.put_nowait(frame)
            with self._lock:
                self._enqueued += 1
        except Full:
            with self._lock:
                self._dropped += 1

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                frame_to_write = self.resize_fn(frame) if self.resize_fn is not None else frame
                if frame_to_write is not None and self.writer is not None:
                    self.writer.write(frame_to_write)
                    with self._lock:
                        self._written += 1
                now = time.time()
                if self._log_interval > 0 and now - self._last_log_time >= self._log_interval:
                    stats = self.snapshot_stats()
                    print(
                        f'[per-id-video] async path={self.path} queued={stats["queued"]} '
                        f'written={stats["written"]} dropped={stats["dropped"]} '
                        f'errors={stats["write_errors"]}'
                    )
                    self._last_log_time = now
            except Exception as exc:
                with self._lock:
                    self._write_errors += 1
                print(f'[per-id-video] async write failed for {self.path}: {exc}')
            finally:
                self._queue.task_done()

    def snapshot_stats(self):
        with self._lock:
            return {
                'queued': self._queue.qsize(),
                'queue_size': self.queue_size,
                'enqueued': self._enqueued,
                'written': self._written,
                'dropped': self._dropped,
                'write_errors': self._write_errors,
                'backend': self.backend,
                'encoder': self.encoder,
                'path': self.path,
            }

    def release(self):
        if self._released:
            return False
        self._released = True
        self._stop.set()
        try:
            self._thread.join(timeout=30.0)
        except Exception:
            pass
        if self._thread.is_alive():
            print(f'[per-id-video] async writer did not drain before timeout path={self.path}')
        finalized = False
        if self.writer is not None:
            try:
                finalized = bool(self.writer.release())
            except Exception:
                finalized = False
            self.writer = None
        return finalized


def _safe_release_writer(writer):
    if writer is None or not hasattr(writer, 'release'):
        return
    try:
        writer.release()
    except Exception:
        pass


def create_h264_video_writer(path, width, height, fps):
    attempt_order = []
    meta = {
        'writer_mode': 'none',
        'writer_backend': 'none',
        'fallback_used': False,
        'fallback_reason': '',
        'attempt_order': attempt_order,
    }

    attempt_order.append('ffmpeg_hw')
    writer = FfmpegH264Writer(path, width, height, fps, encoders=FFMPEG_HW_ENCODERS)
    if writer.is_opened():
        meta['writer_mode'] = 'hw'
        meta['writer_backend'] = 'ffmpeg'
        return writer, meta
    _safe_release_writer(writer)

    meta['fallback_reason'] = 'ffmpeg_hw_open_failed'
    return None, meta

def parse_core_mask(text: str):
    if text is None:
        return None
    s = str(text).strip().lower()
    if s in ('', 'auto', 'default'):
        return None
    if s == 'all':
        return 0b111
    bits = 0
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                a = int(a); b = int(b)
            except ValueError:
                continue
            for k in range(min(a, b), max(a, b) + 1):
                if 0 <= k <= 2:
                    bits |= (1 << k)
        else:
            try:
                k = int(part, 0)
            except ValueError:
                continue
            if 0 <= k <= 2:
                bits |= (1 << k)
    return bits or None


def core_mask_to_indices(mask):
    if mask is None:
        return []
    indices = []
    for idx in range(3):
        if int(mask) & (1 << idx):
            indices.append(idx)
    return indices


def indices_to_core_mask(indices):
    bits = 0
    for idx in indices or []:
        try:
            core_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= core_idx <= 2:
            bits |= (1 << core_idx)
    return bits or None


def resolve_worker_core_masks(core_mask, workers, strategy='auto'):
    worker_count = max(1, int(workers or 1))
    mode = str(strategy or 'auto').strip().lower()
    if mode not in ('auto', 'share', 'split'):
        mode = 'auto'
    core_indices = core_mask_to_indices(core_mask)
    if not core_indices:
        return [None] * worker_count
    if mode == 'share':
        return [core_mask] * worker_count
    if mode == 'auto':
        mode = 'split' if (worker_count > 1 and len(core_indices) > 1) else 'share'
    if mode == 'share':
        return [core_mask] * worker_count
    return [indices_to_core_mask([core_indices[i % len(core_indices)]]) for i in range(worker_count)]


def resolve_auto_plate_core_mask(requested_plate_mask, main_core_mask, worker_core_masks, reserved_masks=None):
    if requested_plate_mask is not None:
        return requested_plate_mask
    main_indices = set(core_mask_to_indices(main_core_mask))
    if not main_indices:
        return None
    used_indices = set()
    for mask in worker_core_masks or []:
        used_indices.update(core_mask_to_indices(mask))
    for mask in reserved_masks or []:
        used_indices.update(core_mask_to_indices(mask))
    spare = sorted(main_indices - used_indices)
    if not spare:
        all_spare = sorted(set((0, 1, 2)) - used_indices)
        spare = all_spare
    if not spare:
        return None
    return indices_to_core_mask([spare[0]])


def _format_ffmpeg_capture_options(options):
    chunks = []
    for key, value in options:
        if value in (None, ''):
            continue
        chunks.append(f'{key};{value}')
    return '|'.join(chunks)


def _merge_ffmpeg_capture_options(extra):
    current = str(os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', '') or '').strip()
    if current and extra:
        return f'{current}|{extra}'
    return extra or current


@contextmanager
def _temporary_env_var(name, value):
    had_original = name in os.environ
    original = os.environ.get(name)
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if had_original:
            os.environ[name] = original
        else:
            os.environ.pop(name, None)


def _open_ffmpeg_capture(src, option_text=''):
    merged_options = _merge_ffmpeg_capture_options(option_text)
    with _temporary_env_var('OPENCV_FFMPEG_CAPTURE_OPTIONS', merged_options):
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)


def _ordered_ffmpeg_hw_decoders(src):
    if not isinstance(src, str):
        return FFMPEG_HW_DECODER_CANDIDATES
    lower = src.lower()
    if '265' in lower or 'hevc' in lower:
        return ('hevc_rkmpp',) + tuple(dec for dec in FFMPEG_HW_DECODER_CANDIDATES if dec != 'hevc_rkmpp')
    return FFMPEG_HW_DECODER_CANDIDATES


def _open_ffmpeg_hardware_capture(src, rtsp_latency_ms=200, rtsp_appsink_max_buffers=1, read_timeout_seconds=5.0):
    if not isinstance(src, str):
        return None
    stream_info = _probe_ffmpeg_stream(src)
    decoder_name = FFMPEG_CODEC_TO_RKMPP_DECODER.get(str((stream_info or {}).get('codec_name') or '').strip().lower(), '')
    if decoder_name:
        cap = FfmpegRawVideoCapture(
            src,
            decoder_name,
            stream_info,
            rtsp_latency_ms=rtsp_latency_ms,
            read_timeout_seconds=read_timeout_seconds,
        )
        if _is_capture_opened(cap):
            print(f'[reader] Using FFmpeg rawvideo hardware decoder {decoder_name} for {src}')
            return cap
        _safe_release_capture(cap)

    base_options = []
    if src.startswith(('rtsp://', 'rtsps://')):
        base_options.append(('rtsp_transport', 'tcp'))
    for decoder in _ordered_ffmpeg_hw_decoders(src):
        option_text = _format_ffmpeg_capture_options(base_options + [
            ('hw_decoders_any', 'rkmpp'),
            ('video_codec', decoder),
        ])
        cap = _open_ffmpeg_capture(src, option_text=option_text)
        if _is_capture_opened(cap):
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            print(f'[reader] Using FFmpeg hardware decoder {decoder} for {src}')
            return TimedVideoCapture(
                cap,
                read_timeout_seconds=read_timeout_seconds,
                backend=f'ffmpeg_hw:{decoder}',
                source=src,
            )
        _safe_release_capture(cap)
    print(f'[reader] ffmpeg hardware decode open failed, source={src}')
    return None


def _open_gstreamer_hardware_capture(
    src,
    rtsp_latency_ms=200,
    rtsp_appsink_max_buffers=1,
    read_timeout_seconds=5.0,
    bgr_mode='direct',
):
    if not isinstance(src, str):
        return None
    pipelines = []
    if src.startswith(('rtsp://', 'rtsps://')):
        rtsp_latency_ms = max(0, int(rtsp_latency_ms))
        rtsp_appsink_max_buffers = max(1, int(rtsp_appsink_max_buffers))
        stream_info = _probe_ffmpeg_stream(src) or {}
        codec_name = str(stream_info.get('codec_name') or '').strip().lower()
        if codec_name in ('hevc', 'h265'):
            depay = 'rtph265depay'
            parser = 'h265parse'
        else:
            depay = 'rtph264depay'
            parser = 'h264parse'
        direct_pipeline = (
            f'rtspsrc location="{src}" latency={rtsp_latency_ms} protocols=tcp ! '
            f'{depay} ! {parser} config-interval=-1 ! '
            'mppvideodec format=BGR ! video/x-raw,format=BGR ! '
            f'appsink sync=false drop=true max-buffers={rtsp_appsink_max_buffers}',
            '[reader] Using GStreamer+mpp direct-BGR RTSP TCP pipeline for {src}',
        )
        safe_pipeline = (
            f'rtspsrc location="{src}" latency={rtsp_latency_ms} protocols=tcp ! '
            f'{depay} ! {parser} config-interval=-1 ! '
            'mppvideodec ! videoconvert ! video/x-raw,format=BGR ! '
            f'appsink sync=false drop=true max-buffers={rtsp_appsink_max_buffers}',
            '[reader] Using GStreamer+mpp safe-BGR RTSP TCP pipeline for {src}',
        )
        pipelines.extend([safe_pipeline, direct_pipeline] if _gstreamer_bgr_mode(bgr_mode) == 'safe' else [direct_pipeline, safe_pipeline])
    elif not src.startswith(('http://', 'https://')):
        stream_info = _probe_ffmpeg_stream(src) or {}
        codec_name = str(stream_info.get('codec_name') or '').strip().lower()
        parser = 'h265parse' if codec_name in ('hevc', 'h265') else 'h264parse'
        direct_pipeline = (
            f'filesrc location="{src}" ! qtdemux ! {parser} ! '
            'mppvideodec format=BGR ! video/x-raw,format=BGR ! appsink',
            '[reader] Using GStreamer+mpp direct-BGR file pipeline for {src}',
        )
        safe_pipeline = (
            f'filesrc location="{src}" ! qtdemux ! {parser} ! '
            'mppvideodec ! videoconvert ! video/x-raw,format=BGR ! appsink',
            '[reader] Using GStreamer+mpp safe-BGR file pipeline for {src}',
        )
        pipelines.extend([safe_pipeline, direct_pipeline] if _gstreamer_bgr_mode(bgr_mode) == 'safe' else [direct_pipeline, safe_pipeline])
    for pipeline, msg in pipelines:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if _is_capture_opened(cap):
            print(msg.format(src=src))
            return TimedVideoCapture(
                cap,
                read_timeout_seconds=read_timeout_seconds,
                backend='gstreamer_hw',
                source=src,
            )
        _safe_release_capture(cap)
    print(f'[reader] gstreamer hardware decode open failed, source={src}')
    return None


def open_video_capture(src, hw_decode=False, rtsp_latency_ms=200, rtsp_appsink_max_buffers=1, read_timeout_seconds=5.0):
    if hw_decode:
        cap = _open_gstreamer_hardware_capture(
            src,
            rtsp_latency_ms=rtsp_latency_ms,
            rtsp_appsink_max_buffers=rtsp_appsink_max_buffers,
            read_timeout_seconds=read_timeout_seconds,
        )
        if _is_capture_opened(cap):
            return cap
        _safe_release_capture(cap)
        cap = _open_ffmpeg_hardware_capture(
            src,
            rtsp_latency_ms=rtsp_latency_ms,
            rtsp_appsink_max_buffers=rtsp_appsink_max_buffers,
            read_timeout_seconds=read_timeout_seconds,
        )
        if _is_capture_opened(cap):
            return cap
        _safe_release_capture(cap)
        return None
    print(f'[reader] hardware decode is disabled, source={src}')
    return None


def _source_kind_for_decode(path):
    if not isinstance(path, str):
        return 'other'
    text = path.strip()
    lower = text.lower()
    if lower.startswith(('rtsp://', 'rtsps://')):
        return 'rtsp'
    if '://' in lower:
        return 'other'
    try:
        candidate = Path(text).expanduser()
        if candidate.is_file():
            return 'file'
    except OSError:
        pass
    return 'file'


def _is_capture_opened(cap):
    if cap is None or not hasattr(cap, 'isOpened'):
        return False
    try:
        return bool(cap.isOpened())
    except Exception:
        return False


def _safe_release_capture(cap):
    if cap is None or not hasattr(cap, 'release'):
        return
    try:
        cap.release()
    except Exception:
        pass


def create_video_reader(path, args):
    """Create capture and return (capture, decode_meta)."""

    def _safe_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    hw = bool(getattr(args, 'hw_decode', False))
    config = getattr(args, '_config', {}) or {}
    video_cfg = config.get('video', {}) or {}
    decode_backend = str(video_cfg.get('decode_backend', 'auto') or 'auto').strip().lower()
    if decode_backend not in {'auto', 'ffmpeg', 'gstreamer'}:
        decode_backend = 'auto'
    rtsp_latency_ms = _safe_int(video_cfg.get('rtsp_latency_ms', 200), 200)
    rtsp_appsink_max_buffers = _safe_int(video_cfg.get('rtsp_appsink_max_buffers', 1), 1)
    gstreamer_bgr_mode = _gstreamer_bgr_mode(video_cfg.get('gstreamer_bgr_mode', 'direct'))
    try:
        reader_frame_timeout_seconds = float(video_cfg.get('reader_frame_timeout_seconds', 5.0))
    except (TypeError, ValueError):
        reader_frame_timeout_seconds = 5.0
    reader_frame_timeout_seconds = max(0.0, reader_frame_timeout_seconds)
    decode_meta = {
        'decode_mode': 'none',
        'decode_backend': 'none',
        'fallback_used': False,
        'fallback_reason': '',
        'source_kind': _source_kind_for_decode(path),
        'attempt_order': [],
        'reader_frame_timeout_seconds': reader_frame_timeout_seconds,
        'requested_backend': decode_backend,
    }

    attempt_order = decode_meta['attempt_order']
    if not hw:
        decode_meta['fallback_reason'] = 'hardware_decode_disabled'
        return None, decode_meta

    if hw:
        if decode_backend in {'auto', 'gstreamer'}:
            attempt_order.append('gstreamer_hw')
            cap_hw = _open_gstreamer_hardware_capture(
                path,
                rtsp_latency_ms=rtsp_latency_ms,
                rtsp_appsink_max_buffers=rtsp_appsink_max_buffers,
                read_timeout_seconds=reader_frame_timeout_seconds,
                bgr_mode=gstreamer_bgr_mode,
            )
            if _is_capture_opened(cap_hw):
                decode_meta['decode_mode'] = 'hw'
                decode_meta['decode_backend'] = 'gstreamer'
                return cap_hw, decode_meta
            _safe_release_capture(cap_hw)
            if decode_backend == 'gstreamer':
                decode_meta['fallback_used'] = True
                decode_meta['fallback_reason'] = 'gstreamer_hw_open_failed'
        if decode_backend in {'auto', 'ffmpeg'}:
            attempt_order.append('ffmpeg_hw')
            cap_hw = _open_ffmpeg_hardware_capture(
                path,
                rtsp_latency_ms=rtsp_latency_ms,
                rtsp_appsink_max_buffers=rtsp_appsink_max_buffers,
                read_timeout_seconds=reader_frame_timeout_seconds,
            )
            if _is_capture_opened(cap_hw):
                decode_meta['decode_mode'] = 'hw'
                decode_meta['decode_backend'] = 'ffmpeg'
                if decode_backend == 'auto':
                    decode_meta['fallback_used'] = True
                    decode_meta['fallback_reason'] = 'gstreamer_hw_open_failed'
                return cap_hw, decode_meta
            _safe_release_capture(cap_hw)
            if decode_backend == 'ffmpeg':
                decode_meta['fallback_used'] = True
                decode_meta['fallback_reason'] = 'ffmpeg_hw_open_failed'

    decode_meta['decode_mode'] = 'none'
    decode_meta['decode_backend'] = 'none'
    decode_meta['fallback_used'] = bool(decode_backend == 'auto' and len(attempt_order) > 1)
    if not decode_meta['fallback_reason']:
        if decode_backend == 'ffmpeg':
            decode_meta['fallback_reason'] = 'ffmpeg_hw_open_failed'
        elif decode_backend == 'gstreamer':
            decode_meta['fallback_reason'] = 'gstreamer_hw_open_failed'
        else:
            decode_meta['fallback_reason'] = 'hardware_open_failed'
    return None, decode_meta


def _per_id_has_valid_plate_candidate(track_state, event_manager):
    checker = getattr(event_manager, '_has_valid_plate_candidate', None)
    if callable(checker):
        try:
            return bool(checker(track_state or {}))
        except Exception:
            return False
    state = track_state or {}
    return bool(
        state.get('plate_text_locked')
        or state.get('plate_candidate_latest')
        or state.get('plate_text_latest')
        or state.get('plate_text')
    )


def emit_per_id_video_type6(track_id, track_state, event_manager, per_id_video_enabled):
    state = track_state or {}
    if state.get('per_id_type6_emitted'):
        return False
    keep_video = bool(state.get('type2_qualified'))
    if not keep_video:
        return False
    require_plate_candidate = bool(getattr(event_manager, 'per_id_type6_require_plate_candidate', False))
    if require_plate_candidate and not _per_id_has_valid_plate_candidate(state, event_manager):
        print(f'[per-id-video] suppress type6 without valid plate candidate: track={track_id}')
        return False

    frame_idx = state.get('record_stop_frame')
    if frame_idx is None:
        frame_idx = state.get('last_frame_idx', 0)
    frame = state.get('last_frame')
    try:
        event_manager.emit_event(
            track_id,
            6,
            frame_idx,
            frame,
            {'perIdVideoEnabled': bool(per_id_video_enabled)},
            state,
        )
    except Exception:
        return False
    state['per_id_type6_emitted'] = True
    return True


def finalize_per_id_recording(writer, track_id, track_state, event_manager, per_id_video_enabled=True):
    if writer is None:
        return False
    finalized = False
    try:
        finalized = bool(writer.release())
    except Exception:
        finalized = False
    if not finalized:
        return False

    state = track_state or {}
    output_path = Path(getattr(writer, 'path', '') or '')
    keep_video = bool(state.get('type2_qualified'))
    if not keep_video:
        if DELETE_UNQUALIFIED_PER_ID_VIDEO and output_path:
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
        return False

    if output_path and not output_path.exists():
        return False
    return emit_per_id_video_type6(track_id, state, event_manager, per_id_video_enabled)


def detect_source_mode(path, override='auto', base_dir=None):
    mode = (override or 'auto').lower()
    if isinstance(path, str):
        lower = path.lower().strip()
        if lower.startswith(('rtsp://', 'rtsp:', 'rtmp://', 'rtp://', 'rtsps://', 'http://', 'https://')):
            return 'camera'
        try:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                root = Path(base_dir) if base_dir is not None else Path.cwd()
                candidate = (root / candidate).resolve()
            if candidate.is_file():
                return 'file'
        except OSError:
            pass
    if mode == 'file':
        return 'file'
    return 'camera'


def _resolve_runtime_path(path_value, base_dir):
    if path_value in (None, ''):
        return None
    if isinstance(path_value, Path):
        path = path_value
    else:
        path = Path(str(path_value))
    path = path.expanduser()
    if not path.is_absolute():
        base = base_dir if base_dir else Path.cwd()
        path = base / path
    return path.resolve()


def _collect_storage_directories(config, base_dir):
    directories = []

    def push(value, treat_as_file=False):
        resolved = _resolve_runtime_path(value, base_dir)
        if not resolved:
            return
        directories.append(resolved.parent if treat_as_file else resolved)

    video_cfg = config.get('video', {}) or {}
    base = base_dir if base_dir else Path.cwd()
    directories.append((base / 'video_result').resolve())

    push(config.get('event_capture_dir'))
    directories.append((base / 'captures').resolve())

    push(config.get('event_output_dir'))
    directories.append((base / 'events').resolve())

    push(video_cfg.get('csv'), treat_as_file=True)
    debug_frame = video_cfg.get('debug_frame_path')
    if debug_frame:
        debug_path = _resolve_runtime_path(debug_frame, base_dir)
        if debug_path:
            directories.append(debug_path.parent if debug_path.suffix else debug_path)
    unique = []
    seen = set()
    for directory in directories:
        if not directory:
            continue
        resolved = Path(directory).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique
