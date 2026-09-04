from collections.abc import Callable
from typing import Any

EventSink = Callable[[str, dict[str, Any]], None]


class EventEmitter:
    """Emits SSE events to a configured sink, or acts as a no-op in headless mode.
    
    Also records emitted events for testing and introspection.
    """

    def __init__(self, sink: EventSink | None = None) -> None:
        self._sink = sink
        self._history: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self._history.append((event, data))
        if self._sink is not None:
            self._sink(event, data)

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        self.emit(event, data)

    @property
    def history(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()


def create_emitter(sink: EventSink | None = None) -> EventEmitter:
    return EventEmitter(sink=sink)
