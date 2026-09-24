class ResultWatchdog:
    """Track consecutive overdue inference results and fail closed after a bounded burst."""

    def __init__(self, max_consecutive=3):
        self.max_consecutive = max(1, int(max_consecutive or 1))
        self.consecutive = 0
        self.total = 0

    def record_timeout(self):
        self.consecutive += 1
        self.total += 1
        return self.consecutive >= self.max_consecutive

    def record_success(self):
        self.consecutive = 0
