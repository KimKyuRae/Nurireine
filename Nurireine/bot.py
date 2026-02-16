"""
Nurireine Discord Bot

Main bot class with AI-powered conversational abilities.
Uses a layered memory system (L1/L2/L3) for context management.
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional

import discord
from discord.ext import commands

from . import config
from .database import DatabaseManager
from .debug_server import get_server, broadcast_event
from .core import MessageHandler, ContextManager
from .ai import Gatekeeper, MemoryManager, MainLLM
from .utils.text import ultra_slim_extract, math_style_compress, replace_user_handles

logger = logging.getLogger(__name__)

# ===========================================================================
# Security Pre-filter Constants (offer.md §5)
# ===========================================================================
_MAX_INPUT_LENGTH = 2000  # Max chars before rejection
_INJECTION_PATTERNS = [
    "이전 지시를 잊어", "관리자 모드", "시스템 프롬프트를 보여",
    "지금부터 ~처럼 행동", "시스템 명령", "시스템 설정을 변경",
    "system prompt", "ignore previous", "ignore all instructions",
    "you are now", "act as", "admin mode", "developer mode",
    "DAN", "jailbreak",
]

# ===========================================================================
# LLM Rate Limiting
# ===========================================================================
_LLM_RATE_LIMIT = 1       # Max LLM calls per window
_LLM_RATE_WINDOW = 20.0   # Window in seconds
_PRIORITY_THRESHOLD = 0.85  # BERT score above this = high priority (always pass)


class Nurireine(commands.Bot):
    """
    Main Discord bot class.
    
    Features:
    - AI-powered conversation with layered memory
    - Message batching and debouncing
    - Active channel tracking per guild
    - Background AI loading for fast startup
    """
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        allowed_mentions = discord.AllowedMentions.none()
        super().__init__(
            command_prefix=config.bot.command_prefix, 
            intents=intents,
            allowed_mentions=allowed_mentions
        )
        
        # Database
        self.db = DatabaseManager()
        
        # AI components (lazy loaded)
        self._ai_loaded = False
        self.gatekeeper: Optional[Gatekeeper] = None
        self.memory: Optional[MemoryManager] = None
        self.llm: Optional[MainLLM] = None
        
        # Message handling
        self._message_handler = MessageHandler(
            self, 
            debounce_delay=config.bot.debounce_delay,
            max_batch_size=config.bot.max_batch_size
        )
        self._context_manager = ContextManager()
        
        # Active channel tracking (Guild ID -> Channel ID)
        self.active_channels: Dict[int, int] = {}
        
        # Performance Tracking
        self.last_stats: Dict[str, float] = {}
        
        # LLM Rate Limiting: sliding window of call timestamps
        self._llm_call_timestamps: List[float] = []
        
        logger.info("Nurireine initialized. AI systems set to lazy load.")
    
    # =========================================================================
    # Lifecycle Events
    # =========================================================================
    
    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        # Load extensions
        await self._load_extensions()
        
        # Start debug server
        await get_server().start()
    
    async def on_ready(self) -> None:
        """Called when the bot is fully connected."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("-" * 40)
        
        # Start background AI loading
        if not self._ai_loaded:
            self.loop.create_task(self._load_ai_systems())
        
        # Start message processing worker
        await self._message_handler.start_worker()
    
    async def close(self) -> None:
        """Called when the bot is shutting down."""
        logger.info("Shutting down Nurireine...")
        
        # Save memory state
        if self.memory:
            logger.info("Saving L2 memory to database...")
            await self.memory.save_all_summaries()
        
        # Stop message handler
        await self._message_handler.stop_worker()
        
        await super().close()
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    async def _load_extensions(self) -> None:
        """
        Load Discord extensions/cogs.
        
        Critical extensions will raise an error on failure.
        Optional extensions will just log a warning.
        """
        # Critical extensions - bot won't function properly without these
        critical_extensions = ["Nurireine.cogs.core"]
        
        # Optional extensions - nice to have but not essential
        optional_extensions = ["jishaku"]
        
        # Load critical extensions first
        for ext in critical_extensions:
            try:
                await self.load_extension(ext)
                logger.info(f"Loaded critical extension: {ext}")
            except Exception as e:
                logger.critical(f"Failed to load critical extension {ext}: {e}")
                raise RuntimeError(f"Critical extension {ext} failed to load: {e}")
        
        # Load optional extensions
        for ext in optional_extensions:
            try:
                await self.load_extension(ext)
                logger.info(f"Loaded optional extension: {ext}")
            except Exception as e:
                logger.warning(f"Failed to load optional extension {ext}: {e} (bot will continue)")
    
    async def _load_ai_systems(self) -> None:
        """Load AI systems in background thread."""
        logger.info("Loading AI systems in background...")
        try:
            await self.loop.run_in_executor(None, self._init_ai_components)
            self._ai_loaded = True
            logger.info("AI Core Systems Online.")
        except Exception as e:
            logger.error(f"Failed to load AI systems: {e}")
    
    def _init_ai_components(self) -> None:
        """Initialize AI components (blocking, run in executor)."""
        logger.info("Initializing AI components...")
        
        self.gatekeeper = Gatekeeper()
        self.memory = MemoryManager(self.gatekeeper, self.db)
        self.llm = MainLLM()
        
        # Restore active channels
        self._restore_active_channels()
    
    def _restore_active_channels(self) -> None:
        """Restore active channels from database."""
        logger.info("Restoring active channels...")
        
        saved = self.db.load_active_channels()
        for guild_id, channel_id, channel_name in saved:
            guild = self.get_guild(guild_id)
            if not guild:
                logger.warning(f"Guild {guild_id} not found (bot may have left).")
                continue
            
            channel = guild.get_channel(channel_id)
            
            if channel:
                self.active_channels[guild_id] = channel.id
                logger.info(f"Restored active channel for '{guild.name}': #{channel.name}")
            else:
                # Try to find by name
                logger.info(f"Channel {channel_id} not found, searching by name '{channel_name}'...")
                found = discord.utils.get(guild.text_channels, name=channel_name)
                
                if found:
                    self.active_channels[guild_id] = found.id
                    self.db.save_active_channel(guild_id, found.id, found.name)
                    logger.info(f"Found alternative: #{found.name}")
                else:
                    logger.warning(f"Could not find channel '{channel_name}' in '{guild.name}'.")
                    self.db.remove_active_channel(guild_id)
    
    # =========================================================================
    # Message Handling
    # =========================================================================
    
    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages."""
        
        # Log to Database
        try:
            log_content = message.content
            # Treat both attachments and stickers as attachments for logging
            has_attachments = bool(message.attachments) or bool(message.stickers)
            is_bot = message.author.bot
            
            # 1. Handle Attachments
            # If msg has attachments or stickers, append/set [ATTACHMENTS]
            if has_attachments:
                if log_content:
                    log_content += " [ATTACHMENTS]"
                else:
                    log_content = "[ATTACHMENTS]"
            
            # 2. Handle Bot Responses
            # If msg is from a bot (including self), prepend [BOT_RESPONSE]
            if is_bot:
                if log_content:
                    log_content = f"[BOT_RESPONSE] {log_content}"
                else:
                    log_content = "[BOT_RESPONSE]"
            
            # Execute Log
            await self.db.async_log_chat_message(
                guild_id=message.guild.id if message.guild else None,
                channel_id=message.channel.id,
                user_id=message.author.id,
                user_name=message.author.name,
                content=log_content,
                is_bot=is_bot,
                has_attachments=has_attachments
            )
        except Exception as e:
            logger.error(f"Error logging message: {e}")

        # Ignore self
        if message.author.id == self.user.id:
            return
        
        # Handle commands
        if message.content.startswith(self.command_prefix):
            await self.process_commands(message)
            return
        
        # Track arrival time (Wall clock)
        self.last_stats["last_msg_arrival"] = message.created_at.timestamp()
        self.last_stats["arrival_wall"] = time.time()
        
        # Queue for AI processing
        await self._message_handler.enqueue_message(message)
    
    async def process_message_batch(self, messages: List[discord.Message]) -> None:
        """
        Process a batch of messages (called by MessageHandler).
        
        Args:
            messages: List of messages to process as a batch
        """
        if not messages:
            return
        
        last_message = messages[-1]
        channel_id = last_message.channel.id
        
        logger.info(f"Processing batch: {len(messages)} msgs in channel {channel_id} from {last_message.author.display_name}")
        
        # Combine message content (with reply context)
        parts = []
        for m in messages:
            text = m.content.strip()
            if not text:
                continue
            
            # If this is a reply, prepend the referenced message content
            if m.reference:
                ref = m.reference.resolved
                # Fallback: fetch from API if not in cache
                if ref is None and m.reference.message_id:
                    try:
                        ref = await m.channel.fetch_message(m.reference.message_id)
                    except Exception:
                        ref = None
                
                if ref:
                    ref_author = getattr(ref.author, 'display_name', '?')
                    ref_content = (ref.content or '').strip()
                    if ref_content:
                        # Truncate long references
                        if len(ref_content) > 100:
                            ref_content = ref_content[:100] + "..."
                        text = f"[{ref_author}의 '{ref_content}'에 대한 답장] {text}"
            
            parts.append(text)
        
        combined = "\n".join(parts)
        if not combined:
            logger.info("Empty combined content, skipping")
            return
        
        # Broadcast user input
        broadcast_event("user_input", {
            "channel_id": channel_id,
            "user_name": last_message.author.display_name,
            "content": combined,
            "timestamp": last_message.created_at.timestamp()
        })
        
        # Start timing
        queue_end_time = time.time() # Wall clock
        self.last_stats["process_start_wall"] = queue_end_time
        
        # Compress content if needed
        processed = math_style_compress(combined)
        if len(processed) > 200:
            processed = ultra_slim_extract(
                processed, 
                config.bot.trigger_keywords, 
                max_final_len=100
            )
        
        # Check if we should respond
        is_explicit = self._context_manager.is_explicit_call(
            messages, self.user, config.bot.call_names
        )
        is_active = self._context_manager.is_active_channel(
            last_message, self.active_channels
        )
        
        # Debug logging for response decision
        logger.info(
            f"Response check - Channel: {channel_id}, "
            f"Explicit: {is_explicit}, Active: {is_active}, "
            f"Guild active_channel: {self.active_channels.get(last_message.guild.id if last_message.guild else None)}"
        )
        
        if not (is_explicit or is_active):
            logger.info(f"Skipping response - not explicit and not active channel")
            return
        
        # Pre-filter input for security (offer.md §5)
        if self._prefilter_input(processed, last_message):
            logger.warning(f"Input rejected by pre-filter in channel {channel_id}")
            return
        
        # Process with AI
        await self._handle_ai_response(
            messages, last_message, channel_id, processed, is_explicit
        )
    
    @staticmethod
    def _prefilter_input(content: str, message: discord.Message) -> bool:
        """
        Rule-based pre-filter for abnormally long or injection-heavy inputs.
        Returns True if the input should be REJECTED (not processed).
        """
        # Check length
        if len(content) > _MAX_INPUT_LENGTH:
            logger.warning(
                f"Input rejected: too long ({len(content)} chars) "
                f"from {message.author.display_name}"
            )
            return True
        
        # Count injection patterns
        content_lower = content.lower()
        injection_count = sum(1 for p in _INJECTION_PATTERNS if p.lower() in content_lower)
        if injection_count >= 3:
            logger.warning(
                f"Input rejected: {injection_count} injection patterns detected "
                f"from {message.author.display_name}"
            )
            return True
        
        return False
    
    def _check_rate_limit(self, is_explicit: bool, analysis: Dict) -> bool:
        """
        Check if an LLM call is allowed under the rate limit.
        
        Priority system:
        - Explicit calls (mentions/call names): ALWAYS pass
        - High BERT score (>0.85): ALWAYS pass (strong engagement signal)
        - Others: Only pass if under rate limit
        
        Returns True if the call is allowed, False if rate-limited.
        """
        now = time.time()
        
        # Purge expired timestamps outside the window
        self._llm_call_timestamps = [
            t for t in self._llm_call_timestamps 
            if now - t < _LLM_RATE_WINDOW
        ]
        
        # Explicit calls always pass
        if is_explicit:
            self._llm_call_timestamps.append(now)
            return True
        
        # High-priority messages (strong BERT confidence) always pass
        perf_stats = analysis.get("_perf_stats", {})
        bert_score = perf_stats.get("bert_score", 0.0)
        if bert_score >= _PRIORITY_THRESHOLD:
            self._llm_call_timestamps.append(now)
            logger.debug(f"High-priority message (BERT={bert_score:.4f}), bypassing rate limit")
            return True
        
        # Standard rate limit check
        if len(self._llm_call_timestamps) >= _LLM_RATE_LIMIT:
            return False
        
        self._llm_call_timestamps.append(now)
        return True
    
    async def _handle_ai_response(
        self,
        messages: List[discord.Message],
        last_message: discord.Message,
        channel_id: int,
        processed_content: str,
        is_explicit_call: bool
    ) -> None:
        """Handle AI processing and response generation."""
        # Auto-switch active channel on explicit call
        if is_explicit_call and last_message.guild:
            await self._handle_channel_switch(messages, last_message)
        
        # Check if AI is ready
        if not self._ai_loaded:
            if is_explicit_call:
                await last_message.reply(
                    "아직 잠에서 깨고 있어요... 잠시만 기다려 주세요! (모델 로딩 중)"
                )
            return
        
        # Run SLM analysis
        context, analysis = await self._run_analysis(
            channel_id, processed_content, last_message, is_explicit=is_explicit_call
        )
        
        # === Interrupt & Merge: Check for new messages that arrived during analysis ===
        new_messages = await self._message_handler.drain_pending_messages(channel_id)
        if new_messages:
            # Filter to same user's messages only
            user_id = last_message.author.id
            same_user_msgs = [m for m in new_messages if m.author.id == user_id]
            other_msgs = [m for m in new_messages if m.author.id != user_id]
            
            if same_user_msgs:
                # Build additional content (with reply context support)
                additional_parts = []
                for m in same_user_msgs:
                    text = m.content.strip()
                    if not text:
                        continue
                    if m.reference:
                        ref = m.reference.resolved
                        if ref is None and m.reference.message_id:
                            try:
                                ref = await m.channel.fetch_message(m.reference.message_id)
                            except Exception:
                                ref = None
                        if ref:
                            ref_author = getattr(ref.author, 'display_name', '?')
                            ref_content = (ref.content or '').strip()
                            if ref_content:
                                if len(ref_content) > 100:
                                    ref_content = ref_content[:100] + "..."
                                text = f"[{ref_author}의 '{ref_content}'에 대한 답장] {text}"
                    additional_parts.append(text)
                
                if additional_parts:
                    merged_addition = "\n".join(additional_parts)
                    processed_content = processed_content + "\n" + merged_addition
                    last_message = same_user_msgs[-1]  # Update to latest message
                    
                    logger.info(
                        f"Merged {len(same_user_msgs)} additional messages during processing. "
                        f"Combined content: '{processed_content[:80]}...'"
                    )
                    
                    # Re-run analysis with merged content
                    context, analysis = await self._run_analysis(
                        channel_id, processed_content, last_message, is_explicit=is_explicit_call
                    )
            
            # Re-queue other users' messages back for separate processing
            if other_msgs:
                await self._message_handler._processing_queue.put(other_msgs)
        
        # Determine if we should respond
        should_respond = is_explicit_call or analysis.get("response_needed", False)
        
        if not should_respond:
            return
        
        # Rate limit check (explicit calls bypass)
        if not self._check_rate_limit(is_explicit_call, analysis):
            logger.info(
                f"LLM call rate-limited for channel {channel_id} "
                f"(non-priority, {len(self._llm_call_timestamps)}/{_LLM_RATE_LIMIT} in window)"
            )
            return
        
        # Generate and send response
        async with last_message.channel.typing():
            await self._generate_and_send_response(
                last_message, channel_id, processed_content, context, is_explicit_call
            )
    
    async def _handle_channel_switch(
        self, 
        messages: List[discord.Message], 
        last_message: discord.Message
    ) -> None:
        """Handle switching the active channel."""
        guild_id = last_message.guild.id
        old_channel_id = self.active_channels.get(guild_id)
        
        if old_channel_id == last_message.channel.id:
            return
        
        # Update active channel
        self.active_channels[guild_id] = last_message.channel.id
        self.db.save_active_channel(
            guild_id, last_message.channel.id, last_message.channel.name
        )
        logger.info(f"[Auto-Switch] Active channel moved to: #{last_message.channel.name}")
        
        # Sync context from new channel
        if self.memory:
            await self._context_manager.sync_channel_history(
                last_message.channel,
                self.memory,
                self.user.id,
                before_message=messages[0]
            )
    
    async def _run_analysis(
        self,
        channel_id: int,
        content: str,
        message: discord.Message,
        is_explicit: bool = False
    ) -> tuple:
        """Run SLM analysis via memory manager."""
        loop = asyncio.get_running_loop()
        slm_start = loop.time()
        
        if not self.memory:
            logger.warning("Memory manager not initialized.")
            return {
                "l3_facts": "",
                "l2_summary": "메모리 시스템이 비활성화되어 있습니다.",
                "l1_recent": []
            }, {}
        
        try:
            guild_id = str(message.guild.id) if message.guild else None
            user_id = str(message.author.id)
            user_name = message.author.display_name
            
            # Use lambda to pass all arguments including user_name
            context, analysis = await self.memory.plan_response(
                channel_id, content, guild_id, user_id, user_name, is_explicit=is_explicit
            )
            
            # Extract detailed stats if available
            if analysis and "_perf_stats" in analysis:
                self.last_stats.update(analysis["_perf_stats"])
            
            self.last_stats["slm_total_duration"] = loop.time() - slm_start
            return context, analysis
            
        except Exception as e:
            logger.error(f"SLM Analysis Error: {e}")
            # Return minimal valid context instead of empty strings
            # This ensures LLM can still generate coherent responses
            return {
                "l3_facts": "",
                "l2_summary": "컨텍스트를 불러오는 중 오류가 발생했습니다. 이전 대화 맥락 없이 응답합니다.",
                "l1_recent": []
            }, {"response_needed": True}
    
    async def _generate_and_send_response(
        self,
        message: discord.Message,
        channel_id: int,
        user_input: str,
        context: Dict,
        is_explicit_call: bool
    ) -> None:
        """Generate LLM response with streaming and send it."""
        if not self.llm:
            return
        
        try:
            # Add user info to context
            context['user_id'] = str(message.author.id)
            context['user_name'] = message.author.display_name
            context['bot_id'] = str(self.user.id) if self.user else 'Unknown'
            context['guild_id'] = str(message.guild.id) if message.guild else None
            
            broadcast_event("llm_generate", {
                "stage": "start", 
                "context_preview": str(context)[:200]
            })
            
            loop = asyncio.get_running_loop()
            llm_start = loop.time()
            
            # Send initial placeholder
            current_message = await message.reply("...")
            full_content = ""
            display_content = ""
            last_edit_time = 0.0
            edit_interval = 1.0  # seconds between edits
            
            # Discord limit safety
            CHUNK_SIZE = 1900 
            
            try:
                # Stream content
                async for chunk in self.llm.generate_response_stream(
                    user_input, context,
                    memory_manager=self.memory,
                    channel_id=channel_id
                ):
                    full_content += chunk
                    
                    # Process content for display
                    temp_processed = replace_user_handles(full_content)
                    if self.user:
                        temp_processed = temp_processed.replace(f"<@{self.user.id}>", "저")
                        temp_processed = temp_processed.replace(f"<@!{self.user.id}>", "저")
                    # Fix unnatural self-reference patterns like '저님'
                    temp_processed = re.sub(r'저님', '저', temp_processed)
                    
                    # Update message periodically
                    current_time = loop.time()
                    if current_time - last_edit_time >= edit_interval:
                        # Only update if content changed and within limit of current message
                        if temp_processed != display_content:
                            if len(temp_processed) <= CHUNK_SIZE:
                                await current_message.edit(content=temp_processed + " ...")
                                display_content = temp_processed
                                last_edit_time = current_time
                            else:
                                # If exceeds limit during stream, we can't easily split mid-stream without spamming
                                # So just stop updating live and wait for finish
                                pass

            except asyncio.TimeoutError:
                 # Handle timeout gracefully
                 full_content += "\n(시간 초과로 중단됨)"
            
            llm_end = loop.time()
            self.last_stats["llm_duration"] = llm_end - llm_start
            
            broadcast_event("llm_generate", {"stage": "end", "response": full_content})
            
            # Final Post-processing
            processed_text = replace_user_handles(full_content)
            if self.user:
                processed_text = processed_text.replace(f"<@{self.user.id}>", "저")
                processed_text = processed_text.replace(f"<@!{self.user.id}>", "저")
            # Fix unnatural self-reference patterns (e.g. '저님', '저씨')
            processed_text = re.sub(r'저님', '저', processed_text)
            processed_text = re.sub(r'저씨', '저', processed_text)
            
            # Save to memory
            if self.memory:
                await self.memory.add_message(
                    channel_id, 
                    "user", 
                    user_input, 
                    user_name=message.author.display_name,
                    user_id=str(message.author.id)
                )
                await self.memory.add_message(channel_id, "assistant", processed_text)
            
            # Final Message Update/Send
            if len(processed_text) <= CHUNK_SIZE:
                if processed_text.strip():
                    await current_message.edit(content=processed_text)
                else:
                    await current_message.edit(content="...")
            else:
                # Split if too long
                chunks = [processed_text[i:i+CHUNK_SIZE] for i in range(0, len(processed_text), CHUNK_SIZE)]
                
                # Edit first message with first chunk
                if chunks:
                    await current_message.edit(content=chunks[0])
                
                # Send remaining chunks as new messages
                for chunk in chunks[1:]:
                    await message.channel.send(chunk)

            broadcast_event("response_sent", {
                "channel_id": channel_id, 
                "content": processed_text
            })
            
            self.last_stats["total_turnaround"] = time.time() - self.last_stats.get("process_start_wall", time.time())
            
        except discord.HTTPException as e:
            logger.error(f"Failed to send/edit message: {e}")
            try:
                # If we fail to edit, try sending new
                await message.channel.send("메시지 전송 중 오류가 발생했습니다.")
            except:
                pass
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            if is_explicit_call:
                await message.reply("죄송해요, 응답을 생성하는 중에 문제가 생겼어요.")