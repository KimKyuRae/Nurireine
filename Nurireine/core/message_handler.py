"""
Message Handler Module

Handles message batching, debouncing, and queue processing.
Separates message collection logic from the main bot class.
"""

import asyncio
import logging
import collections
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, field

import discord

if TYPE_CHECKING:
    from ..bot import Nurireine

logger = logging.getLogger(__name__)


@dataclass
class ChannelQueue:
    """Represents the message queue state for a single channel."""
    messages: List[discord.Message] = field(default_factory=list)
    last_message_time: float = 0.0
    debounce_task: Optional[asyncio.Task] = None
    sequence_number: int = 0  # Track message ordering


class MessageHandler:
    """
    Handles message batching and debouncing for the bot.
    
    Messages are collected per-channel and processed after a debounce delay.
    This allows multiple rapid messages to be processed as a single batch.
    
    Thread Safety:
    - Uses asyncio.Lock for queue modifications to prevent race conditions
    - Sequence numbers ensure proper message ordering
    """
    
    def __init__(self, bot: "Nurireine", debounce_delay: float = 2.0, max_batch_size: int = 10):
        self.bot = bot
        self.debounce_delay = debounce_delay
        self.max_batch_size = max_batch_size
        
        self._channel_queues: Dict[int, ChannelQueue] = {}
        self._queue_locks: Dict[int, asyncio.Lock] = {}  # Per-channel locks
        
        # Processing queue stores the actual BATCH of messages
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        
        # Concurrency Management
        # Tracks which channels are currently being processed by the LLM
        self._active_channels: Set[int] = set()
        # Stores pending batches for channels that are currently busy
        self._pending_batches: Dict[int, collections.deque] = collections.defaultdict(collections.deque)
        # Lock for modifying active_channels and pending_batches
        self._batch_lock = asyncio.Lock()
    
    def _get_channel_lock(self, channel_id: int) -> asyncio.Lock:
        """Get or create a lock for a channel."""
        if channel_id not in self._queue_locks:
            self._queue_locks[channel_id] = asyncio.Lock()
        return self._queue_locks[channel_id]
    
    def _get_channel_queue(self, channel_id: int) -> ChannelQueue:
        """Get or create a channel queue."""
        if channel_id not in self._channel_queues:
            self._channel_queues[channel_id] = ChannelQueue()
        return self._channel_queues[channel_id]
    
    async def start_worker(self) -> None:
        """Start the background worker for processing message batches."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue_worker())
            logger.info("Message processing worker started (Parallel Mode).")
    
    async def stop_worker(self) -> None:
        """Stop the background worker."""
        # Cancel all pending debounce tasks
        for queue in self._channel_queues.values():
            if queue.debounce_task and not queue.debounce_task.done():
                queue.debounce_task.cancel()
        
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Message processing worker stopped.")
    
    async def enqueue_message(self, message: discord.Message) -> None:
        """
        Add a message to the channel's queue and schedule processing.
        
        If the speaker changes, the current batch is immediately flushed to processing.
        This ensures bursts are separated by speaker.
        
        Thread-safe: Uses per-channel locks to prevent race conditions.
        
        Args:
            message: The Discord message to enqueue
        """
        channel_id = message.channel.id
        lock = self._get_channel_lock(channel_id)
        
        async with lock:
            queue = self._get_channel_queue(channel_id)
            
            # Increment sequence number for ordering
            queue.sequence_number += 1
            
            # Check for speaker change
            if queue.messages and queue.messages[-1].author.id != message.author.id:
                logger.debug(f"Speaker changed in channel {channel_id}. Flushing previous batch.")
                
                # Flush the current batch immediately
                batch_to_send = list(queue.messages)
                queue.messages.clear()
                
                # Cancel pending debounce task as we are handling it now
                if queue.debounce_task and not queue.debounce_task.done():
                    queue.debounce_task.cancel()
                
                # Send previous batch to worker
                await self._processing_queue.put(batch_to_send)
            
            # Add new message to queue
            loop = asyncio.get_running_loop()
            queue.last_message_time = loop.time()
            queue.messages.append(message)
            
            # Start or reset debounce timer for this (potentially new) batch
            if queue.debounce_task is None or queue.debounce_task.done():
                queue.debounce_task = asyncio.create_task(
                    self._schedule_processing(channel_id)
                )
    
    async def _schedule_processing(self, channel_id: int) -> None:
        """
        Wait for debounce delay, then submit channel batch for processing.
        Uses adaptive debouncing: increases delay when many channels are concurrently active
        (offer.md §2-나).
        """
        while True:
            loop = asyncio.get_running_loop()
            queue = self._get_channel_queue(channel_id)
            
            if not queue.messages:
                break
            
            # Dynamic Delay Logic
            last_msg = queue.messages[-1]
            content_len = len(last_msg.content.strip())
            
            if content_len < 15:
                target_delay = 3.0
            elif content_len < 50:
                target_delay = 2.0
            else:
                target_delay = 1.0
            
            # Adaptive debouncing: scale delay by concurrent active channels
            active_count = len(self._active_channels)
            if active_count >= 5:
                target_delay = min(target_delay * 3.0, 10.0)
            elif active_count >= 3:
                target_delay = min(target_delay * 2.0, 8.0)
            elif active_count >= 2:
                target_delay = min(target_delay * 1.5, 6.0)
            
            now = loop.time()
            elapsed = now - queue.last_message_time
            remaining = target_delay - elapsed
            
            if remaining <= 0:
                break
            
            await asyncio.sleep(remaining)
        
        # Submit to processing queue if there are messages
        queue = self._get_channel_queue(channel_id) # Re-fetch to be safe
        if queue.messages:
            batch = list(queue.messages)
            queue.messages.clear()
            await self._processing_queue.put(batch)
    
    async def _process_queue_worker(self) -> None:
        """
        Background worker that distributes batches to per-channel tasks.
        Enables parallel processing across different channels while maintaining
        sequential order within each channel.
        """
        while True:
            try:
                # Get the next batch of messages
                messages = await self._processing_queue.get()
                
                if not messages:
                    self._processing_queue.task_done()
                    continue

                channel_id = messages[0].channel.id
                
                async with self._batch_lock:
                    if channel_id in self._active_channels:
                        # Channel is busy, queue it for later execution by the active task
                        self._pending_batches[channel_id].append(messages)
                    else:
                        # Channel is free, mark as active and start processing task
                        self._active_channels.add(channel_id)
                        # We do NOT await here to allow other channels to proceed
                        asyncio.create_task(self._process_batch_task(channel_id, messages))
                
                self._processing_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in processing dispatcher: {e}")

    async def _process_batch_task(self, channel_id: int, initial_messages: List[discord.Message]) -> None:
        """
        Task to process a batch for a specific channel.
        Iteratively processes pending batches for the same channel.
        """
        current_messages = initial_messages
        
        while True:
            try:
                # Limit batch size
                if len(current_messages) > self.max_batch_size:
                    dropped = len(current_messages) - self.max_batch_size
                    logger.warning(
                        f"Batch limit reached for channel {channel_id}. "
                        f"Dropping {dropped} old messages."
                    )
                    current_messages = current_messages[-self.max_batch_size:]
                
                # Process the batch (This is the heavy LLM operation)
                await self.bot.process_message_batch(current_messages)
                
            except Exception as e:
                logger.error(f"Error processing batch for channel {channel_id}: {e}")
                
            # Check for next batch
            async with self._batch_lock:
                if self._pending_batches[channel_id]:
                    current_messages = self._pending_batches[channel_id].popleft()
                else:
                    # No more batches, release the channel
                    if channel_id in self._active_channels:
                        self._active_channels.remove(channel_id)
                    break
    
    def clear_channel(self, channel_id: int) -> None:
        """Clear the message queue for a specific channel."""
        if channel_id in self._channel_queues:
            queue = self._channel_queues[channel_id]
            queue.messages.clear()
            if queue.debounce_task and not queue.debounce_task.done():
                queue.debounce_task.cancel()
    
    async def drain_pending_messages(self, channel_id: int) -> List[discord.Message]:
        """
        Drain all pending/queued messages for a channel.
        
        Collects from two sources:
        1. _pending_batches: Already debounced batches waiting for channel to free up
        2. _channel_queues: Messages still being debounced (cancels the debounce timer)
        
        Returns a flat list of messages, or empty list if nothing pending.
        Used by the bot to merge "단타" (rapid messages) that arrived during processing.
        """
        collected: List[discord.Message] = []
        
        # 1. Drain pending batches (already debounced)
        async with self._batch_lock:
            while self._pending_batches[channel_id]:
                batch = self._pending_batches[channel_id].popleft()
                collected.extend(batch)
        
        # 2. Drain current debounce queue (cancel timer, take messages now)
        lock = self._get_channel_lock(channel_id)
        async with lock:
            queue = self._get_channel_queue(channel_id)
            if queue.messages:
                collected.extend(queue.messages)
                queue.messages.clear()
                # Cancel the debounce timer since we're taking the messages
                if queue.debounce_task and not queue.debounce_task.done():
                    queue.debounce_task.cancel()
                    queue.debounce_task = None
        
        return collected
