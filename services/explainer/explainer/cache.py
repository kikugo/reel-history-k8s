"""In-process LRU cache with a time-to-live. One per pod; nothing shared between replicas."""

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable


def cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class TTLCache:
    def __init__(self, max_items: int = 1024, ttl_s: float = 24 * 3600, clock: Callable[[], float] = time.monotonic):
        self.max_items = max_items
        self.ttl_s = ttl_s
        self.clock = clock
        self._items: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def get(self, key: str) -> str | None:
        hit = self._items.get(key)
        if hit is None:
            return None
        stored_at, value = hit
        if self.clock() - stored_at > self.ttl_s:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        self._items[key] = (self.clock(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
