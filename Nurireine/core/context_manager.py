"""
Context Manager Module

Handles channel context synchronization when the bot switches active channels.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import discord

from .. import config

if TYPE_CHECKING:
    from ..ai.memory import MemoryManager

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages context synchronization when switching between channels.
    
    When the bot's active channel changes, this class loads recent history
    from the new channel to provide context continuity.
    """
    
    def __init__(self, history_limit: int = 50, context_limit: int = 10):
        """
        Args:
            history_limit: Maximum messages to fetch from channel history
            context_limit: Maximum messages to sync into memory
        """
        self.history_limit = history_limit
        self.context_limit = context_limit
    
    async def sync_channel_history(
        self, 
        channel: discord.TextChannel,
        memory: "MemoryManager",
        bot_id: int,
        before_message: Optional[discord.Message] = None
    ) -> bool:
        """
        Synchronize channel history into memory.
        
        Fetches recent messages from the channel and adds them to the
        memory manager's L1 buffer.
        
        Args:
            channel: The Discord channel to sync from
            memory: The memory manager to sync into
            bot_id: The bot's user ID (for role detection)
            before_message: Optional message to fetch history before
            
        Returns:
            True if sync was successful, False otherwise
        """
        try:
            # Fetch raw messages
            raw_messages: List[discord.Message] = []
            async for msg in channel.history(limit=self.history_limit, before=before_message):
                # Filter out bot commands from history
                if not msg.content.startswith(config.bot.command_prefix):
                    raw_messages.append(msg)
            
            raw_messages.reverse()  # Oldest first
            
            # Merge consecutive messages from same author
            merged_history = self._merge_consecutive_messages(raw_messages, bot_id)
            
            # Add to memory (most recent ones)
            channel_id = channel.id
            for msg_data in merged_history[-self.context_limit:]:
                memory.add_message(channel_id, msg_data["role"], msg_data["content"])
            
            logger.info(f"Synced {len(merged_history[-self.context_limit:])} messages from #{channel.name}")
            return True
            
        except discord.Forbidden:
            logger.warning(f"No permission to read history in #{channel.name}")
            return False
        except Exception as e:
            logger.error(f"Failed to sync history from #{channel.name}: {e}")
            return False
    
    def _merge_consecutive_messages(
        self, 
        messages: List[discord.Message], 
        bot_id: int
    ) -> List[Dict[str, str]]:
        """
        Merge consecutive messages from the same author.
        
        Short consecutive messages (< 30 chars) are combined into one.
        
        Args:
            messages: List of Discord messages to merge
            bot_id: The bot's user ID
            
        Returns:
            List of merged message dictionaries with 'role' and 'content'
        """
        merged: List[Dict[str, str]] = []
        current: Optional[Dict[str, str]] = None
        
        for msg in messages:
            content = msg.content.strip()
            if not content:
                continue
            
            role = "assistant" if msg.author.id == bot_id else "user"
            
            # Try to merge with previous message
            if current and current["role"] == role and len(current["content"]) < 30:
                current["content"] += f" {content}"
            else:
                if current:
                    merged.append(current)
                current = {"role": role, "content": content}
        
        if current:
            merged.append(current)
        
        return merged
    
    def is_explicit_call(
        self,
        messages: List[discord.Message],
        bot_user: discord.User,
        call_names: List[str]
    ) -> bool:
        """
        Check if any message in the batch is an explicit bot call.
        
        A message is considered an explicit call if:
        - The bot is @mentioned
        - It's a reply to the bot's message
        - It starts with one of the call names
        
        Args:
            messages: List of messages to check
            bot_user: The bot's Discord user object
            call_names: List of names that trigger the bot
            
        Returns:
            True if any message is an explicit call
        """
        for msg in messages:
            # Check @mention
            if bot_user in msg.mentions:
                return True
            
            # Check reply to bot
            if (msg.reference and 
                msg.reference.resolved and 
                msg.reference.resolved.author.id == bot_user.id):
                return True
            
            # Check call names
            content = msg.content.strip()
            if any(content.startswith(name) for name in call_names):
                return True
        
        return False
    
    def is_active_channel(
        self,
        message: discord.Message,
        active_channels: Dict[int, int]
    ) -> bool:
        """
        Check if the message is in an active (watched) channel.
        
        Args:
            message: The message to check
            active_channels: Mapping of guild_id -> channel_id
            
        Returns:
            True if the message is in an active channel
        """
        if not message.guild:
            return False
        
        guild_id = message.guild.id
        return (
            guild_id in active_channels and 
            active_channels[guild_id] == message.channel.id
        )
