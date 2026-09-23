import time


class WindowedLogThrottler:
    """First message prints immediately; suppressed messages are summarized per window."""

    # key含高基数成分（如entryId）时_states会持续增长，定期清理超过滞留阈值的旧key
    PURGE_INTERVAL_SECONDS = 60.0

    def __init__(self, window_seconds=10.0):
        self.window_seconds = max(0.1, float(window_seconds))
        self._states = {}
        self._last_purge_ts = time.time()

    def record(self, key, message, now=None, window_seconds=None):
        ts = time.time() if now is None else float(now)
        window = self.window_seconds if window_seconds is None else max(0.1, float(window_seconds))
        self._maybe_purge(ts)
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

    def _maybe_purge(self, now):
        if now - self._last_purge_ts < self.PURGE_INTERVAL_SECONDS:
            return
        self._last_purge_ts = now
        horizon = max(self.window_seconds, 1.0) * 4.0
        stale_keys = [
            key
            for key, state in self._states.items()
            if now - float(state.get("window_start", 0.0)) > horizon
        ]
        for key in stale_keys:
            self._states.pop(key, None)


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
