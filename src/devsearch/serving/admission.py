"""Single-slot admission: one active search, no waiting queue (docs/04 "Search API").

The slot is taken *before* work is submitted to the executor and released only by the worker when
its work, including already-submitted device work, has truly finished. A client that disconnects or
an HTTP timeout therefore never frees the slot early. Stdlib only.
"""

import threading


class Admission:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Take the slot without waiting. False means busy."""
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()
