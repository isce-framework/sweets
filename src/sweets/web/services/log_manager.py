"""Log management for streaming job output via WebSocket."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field

# Regex patterns to detect workflow step transitions
STEP_PATTERNS = [
    (1, re.compile(r"(Downloading|Querying ASF|download)", re.IGNORECASE)),
    (2, re.compile(r"(Creating GSLCs|Geocoding|geocode)", re.IGNORECASE)),
    (3, re.compile(r"(Creating.*interferogram|burst interferogram)", re.IGNORECASE)),
    (4, re.compile(r"(Stitching|stitch.*interferogram)", re.IGNORECASE)),
    (5, re.compile(r"(Unwrapping|unwrap)", re.IGNORECASE)),
]


@dataclass
class JobLogBuffer:
    """Buffer for a single job's logs."""

    lines: list[str] = field(default_factory=list)
    current_step: int = 0
    subscribers: set[asyncio.Queue] = field(default_factory=set)

    def append(self, line: str):
        """Add a log line and notify subscribers."""
        self.lines.append(line)

        # Check for step transitions
        for step_num, pattern in STEP_PATTERNS:
            if pattern.search(line) and step_num > self.current_step:
                self.current_step = step_num
                break

        # Notify all subscribers
        for queue in self.subscribers:
            try:
                queue.put_nowait(
                    {"type": "log", "line": line, "step": self.current_step}
                )
            except asyncio.QueueFull:
                pass  # Drop if queue is full

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to log updates. Returns a queue that receives new lines."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from log updates."""
        self.subscribers.discard(queue)

    def get_history(self) -> list[str]:
        """Get all buffered log lines."""
        return list(self.lines)


class LogManager:
    """Manages log buffers for all jobs."""

    def __init__(self):
        self._buffers: dict[int, JobLogBuffer] = defaultdict(JobLogBuffer)
        self._lock = asyncio.Lock()

    def get_buffer(self, job_id: int) -> JobLogBuffer:
        """Get or create a log buffer for a job."""
        return self._buffers[job_id]

    def append_log(self, job_id: int, line: str):
        """Append a log line to a job's buffer."""
        self._buffers[job_id].append(line)

    def clear_buffer(self, job_id: int):
        """Clear a job's log buffer."""
        if job_id in self._buffers:
            del self._buffers[job_id]

    def get_current_step(self, job_id: int) -> int:
        """Get the current workflow step for a job."""
        return self._buffers[job_id].current_step


# Global log manager instance
log_manager = LogManager()
