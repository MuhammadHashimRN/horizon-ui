"""Central pub/sub event bus for inter-layer communication."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable

from horizon.types import Event, EventType

logger = logging.getLogger(__name__)

Callback = Callable[[Event], None]


class EventBus:
    """Thread-safe publish/subscribe event bus.

    Modules subscribe to specific event types and receive callbacks
    when events of that type are published. All callbacks are invoked
    synchronously on the publisher's thread.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callback]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callback) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug("Subscribed %s to %s", callback.__qualname__, event_type.name)

    def unsubscribe(self, event_type: EventType, callback: Callback) -> None:
        with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug("Unsubscribed %s from %s", callback.__qualname__, event_type.name)
            except ValueError:
                pass

    def publish(self, event: Event) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(event.type, []))

        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "Error in subscriber %s for event %s",
                    callback.__qualname__,
                    event.type.name,
                )

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
