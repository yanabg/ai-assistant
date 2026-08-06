"""Bridge an asyncio event loop into tkinter's single-threaded world.

tkinter must run on the main thread, so all coroutine work is offloaded to a
dedicated loop thread. Coroutines are scheduled with :meth:`AsyncLoopThread.submit`
and their results are marshalled back to the UI through a plain, thread-safe
:class:`queue.Queue` that the UI polls with ``after``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import TypeVar

_T = TypeVar("_T")


class AsyncLoopThread:
    """Own an asyncio event loop running on a dedicated daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="chatapp-asyncio",
            daemon=True,
        )

    def start(self) -> None:
        """Start the loop thread and block until it is ready to accept work."""
        self._thread.start()
        self._ready.wait()

    def submit(self, coro: Coroutine[object, object, _T]) -> Future[_T]:
        """Schedule ``coro`` on the loop thread.

        The returned :class:`concurrent.futures.Future` can be cancelled from
        the UI thread; cancellation is propagated to the underlying task.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self, timeout: float = 2.0) -> None:
        """Cancel outstanding work and stop the loop thread (idempotent)."""
        if not self._thread.is_alive():
            return
        shutdown = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            shutdown.result(timeout=timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _shutdown(self) -> None:
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks(self._loop) if task is not current]
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
