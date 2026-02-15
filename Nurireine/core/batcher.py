
import asyncio
import logging
from typing import List, Callable, TypeVar, Generic, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar("T")

class MessageBatcher(Generic[T]):
    """
    A generic async batcher that collects items and processes them in batches
    after a debounce delay.
    """
    def __init__(
        self, 
        callback: Callable[[List[T]], Awaitable[None]], 
        debounce_delay: float = 1.0
    ):
        self.callback = callback
        self.debounce_delay = debounce_delay
        self._queue: List[T] = []
        self._timer: asyncio.Task = None
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self):
        """Start the batcher (no-op in this implementation as it's event-driven)."""
        self._running = True

    async def stop(self):
        """Stop the batcher and cancel pending tasks."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            try:
                await self._timer
            except asyncio.CancelledError:
                pass
            self._timer = None

    async def enqueue(self, item: T):
        """Add an item to the batch and reset the debounce timer."""
        if not self._running:
            logger.warning("Batcher is not running.")
            return

        async with self._lock:
            self._queue.append(item)
            
            # Reset the timer
            if self._timer:
                self._timer.cancel()
            
            self._timer = asyncio.create_task(self._process_after_delay())

    async def _process_after_delay(self):
        """Wait for debounce delay then process the batch."""
        try:
            await asyncio.sleep(self.debounce_delay)
            await self._flush()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in batch timer: {e}")

    async def _flush(self):
        """Flush the queue and call the callback."""
        async with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
            self._timer = None
        
        try:
            await self.callback(batch)
        except Exception as e:
            logger.error(f"Error in batch callback: {e}")
