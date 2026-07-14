"""
A tiny publish/subscribe event bus.

Cameras run in background OS threads while the API and WebSocket handlers run on
the asyncio event loop. This bus bridges the two: threads call
:meth:`EventBus.publish_threadsafe`, which hops onto the loop and fans the event
out to every subscribed :class:`asyncio.Queue`. Slow subscribers drop their
oldest event rather than blocking a camera thread.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class EventBus:
    def __init__(self, max_queue: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_queue = max_queue

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the running event loop; required before threads can publish."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event: dict[str, Any]) -> None:
        """Deliver an event. Must be called on the event loop thread."""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event to make room for the newest.
                try:
                    queue.get_nowait()
                except Exception:
                    pass
                try:
                    queue.put_nowait(event)
                except Exception:
                    pass

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Deliver an event from any thread (used by camera capture threads)."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self.publish, event)
        except RuntimeError:
            # Loop is closed / shutting down.
            pass
