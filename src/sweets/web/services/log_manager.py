"""Log management for streaming job output via WebSocket.

Producer/consumer model: the executor runs the workflow subprocess in a
background thread and pushes each stdout line through ``append_log``.
WebSocket consumers subscribe to a per-job ``asyncio.Queue``. Because the
producer thread is *not* on the asyncio event loop, we route every queue
write through ``loop.call_soon_threadsafe`` so the queue's internal
``_wakeup`` machinery stays loop-bound.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field

# Best-effort regex map from log lines to a 1-5 step bucket. Patterns are
# matched against pipeline log output that we don't control (dolphin /
# sweets.core / COMPASS subprocesses), so this is heuristic — the UI uses
# `max(current_step, this)` and we never go backward, so a missed transition
# only delays the indicator, never corrupts it.
STEP_PATTERNS = [
    (1, re.compile(r"(Downloading|Querying ASF|download)", re.IGNORECASE)),
    (2, re.compile(r"(Creating GSLCs|Geocoding|geocode)", re.IGNORECASE)),
    (3, re.compile(r"(Creating.*interferogram|burst interferogram)", re.IGNORECASE)),
    (4, re.compile(r"(Stitching|stitch.*interferogram)", re.IGNORECASE)),
    (5, re.compile(r"(Unwrapping|unwrap)", re.IGNORECASE)),
]


@dataclass(eq=False)
class _Subscriber:
    """A single WebSocket consumer pinned to the event loop that owns it."""

    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop


@dataclass
class JobLogBuffer:
    """Buffer for a single job's logs."""

    lines: list[str] = field(default_factory=list)
    current_step: int = 0
    subscribers: set[_Subscriber] = field(default_factory=set)
    # Plain threading lock — `append` is called from the executor thread
    # and `subscribe`/`unsubscribe` from the asyncio loop thread, so we
    # need cross-thread safety, not asyncio cooperation.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, line: str):
        """Add a log line and notify subscribers.

        Safe to call from any thread; queue writes are dispatched onto
        each subscriber's owning event loop via ``call_soon_threadsafe``.
        """
        self.lines.append(line)

        for step_num, pattern in STEP_PATTERNS:
            if pattern.search(line) and step_num > self.current_step:
                self.current_step = step_num
                break

        msg = {"type": "log", "line": line, "step": self.current_step}
        with self._lock:
            subs = list(self.subscribers)
        for sub in subs:
            sub.loop.call_soon_threadsafe(_safe_put, sub.queue, msg)

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to log updates. Must be called from inside the event loop."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        sub = _Subscriber(queue=queue, loop=asyncio.get_running_loop())
        with self._lock:
            self.subscribers.add(sub)
        # Stash the wrapper on the queue so unsubscribe doesn't have to
        # rescan the whole subscriber set.
        queue._sweets_sub = sub  # type: ignore[attr-defined]
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from log updates."""
        sub = getattr(queue, "_sweets_sub", None)
        if sub is None:
            return
        with self._lock:
            self.subscribers.discard(sub)

    def get_history(self) -> list[str]:
        """Get all buffered log lines."""
        return list(self.lines)


def _safe_put(queue: asyncio.Queue, msg: dict) -> None:
    """put_nowait, swallowing QueueFull so a stalled client can't OOM us."""
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        pass


class LogManager:
    """Manages log buffers for all jobs."""

    def __init__(self):
        self._buffers: dict[int, JobLogBuffer] = defaultdict(JobLogBuffer)

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
