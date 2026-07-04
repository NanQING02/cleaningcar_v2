import time


class WindowedLogThrottler:
    """First message prints immediately; suppressed messages are summarized per window."""

    def __init__(self, window_seconds=10.0):
        self.window_seconds = max(0.1, float(window_seconds))
        self._states = {}

    def record(self, key, message, now=None, window_seconds=None):
        ts = time.time() if now is None else float(now)
        window = self.window_seconds if window_seconds is None else max(0.1, float(window_seconds))
        state = self._states.get(key)
        if state is None:
            self._states[key] = {"window_start": ts, "suppressed": 0}
            return [message]

        elapsed = ts - float(state["window_start"])
        if elapsed < window:
            state["suppressed"] = int(state["suppressed"]) + 1
            return []

        messages = []
        suppressed = int(state["suppressed"])
        if suppressed > 0:
            messages.append(
                f"[throttle] {key}: suppressed {suppressed} similar messages in {elapsed:.1f}s"
            )
        messages.append(message)
        state["window_start"] = ts
        state["suppressed"] = 0
        return messages


class WindowedLogThrottle:
    """Adapter for callsites that emit directly through a callback."""

    def __init__(self, now_fn=None):
        self._now_fn = now_fn or time.time
        self._throttler = WindowedLogThrottler()

    def log(self, key, message, window_seconds, emit=print):
        now = float(self._now_fn())
        for line in self._throttler.record(
            key=key,
            message=message,
            now=now,
            window_seconds=window_seconds,
        ):
            emit(line)
