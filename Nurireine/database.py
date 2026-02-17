"""
Database Manager Module

Handles SQLite database operations for persistent storage.
Supports both sync and async operations.
"""

import asyncio
import sqlite3
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations.
    
    Tables:
    - active_channels: Per-guild active channel settings
    - channel_summaries: L2 memory summaries per channel
    
    Provides both sync (for blocking contexts) and async (for event loops) methods.
    """
    
    def __init__(self, db_path: Path = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.DATABASE_PATH
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def _init_db(self) -> None:
        """Initialize database tables and run migrations."""
        # Run migrations first
        from .migrations import run_auto_migration
        
        try:
            migration_success = run_auto_migration(self.db_path)
            if not migration_success:
                logger.warning("event=migration_warning message='Migrations completed with warnings'")
        except Exception as e:
            logger.error(f"event=migration_error error={str(e)}")
            # Continue with initialization even if migration fails
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Active channels table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS active_channels (
                        guild_id INTEGER PRIMARY KEY,
                        channel_id INTEGER NOT NULL,
                        channel_name TEXT
                    )
                """)
                
                # Channel summaries table (L2 memory)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS channel_summaries (
                        channel_id INTEGER PRIMARY KEY,
                        summary_text TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Chat logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        guild_id INTEGER,
                        channel_id INTEGER,
                        user_id INTEGER,
                        user_name TEXT,
                        content TEXT,
                        is_bot BOOLEAN,
                        has_attachments BOOLEAN,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Channel policies table (added via migration 002)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS channel_policies (
                        channel_id INTEGER PRIMARY KEY,
                        response_mode TEXT DEFAULT 'balanced',
                        mood_adjustment BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CHECK(response_mode IN ('strict', 'balanced', 'chatty'))
                    )
                """)
                
                # Migration: Add last_updated column if it doesn't exist (for old databases)
                cursor.execute("PRAGMA table_info(channel_summaries)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'last_updated' not in columns:
                    logger.info("event=legacy_migration message='Adding last_updated column to channel_summaries'")
                    # SQLite doesn't allow non-constant defaults in ALTER TABLE
                    # So we add nullable column first, then update existing rows
                    cursor.execute("""
                        ALTER TABLE channel_summaries 
                        ADD COLUMN last_updated TIMESTAMP
                    """)
                    # Set current timestamp for existing rows
                    cursor.execute("""
                        UPDATE channel_summaries 
                        SET last_updated = CURRENT_TIMESTAMP 
                        WHERE last_updated IS NULL
                    """)
                
                conn.commit()
            logger.info("event=database_initialized")
        except Exception as e:
            logger.error(f"event=database_init_failed error={str(e)}")
            raise
    
    # =========================================================================
    # Active Channels (Sync)
    # =========================================================================
    
    def save_active_channel(
        self, 
        guild_id: int, 
        channel_id: int, 
        channel_name: str
    ) -> None:
        """
        Save active channel for a guild (sync).
        
        Args:
            guild_id: Discord guild ID
            channel_id: Discord channel ID
            channel_name: Channel name (for recovery)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO active_channels 
                    (guild_id, channel_id, channel_name)
                    VALUES (?, ?, ?)
                """, (guild_id, channel_id, channel_name))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save active channel: {e}")
    
    def remove_active_channel(self, guild_id: int) -> None:
        """
        Remove active channel setting for a guild (sync).
        
        Args:
            guild_id: Discord guild ID
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM active_channels WHERE guild_id = ?", 
                    (guild_id,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to remove active channel: {e}")
    
    def load_active_channels(self) -> List[Tuple[int, int, str]]:
        """
        Load all active channel settings (sync).
        
        Returns:
            List of (guild_id, channel_id, channel_name) tuples
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT guild_id, channel_id, channel_name FROM active_channels"
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to load active channels: {e}")
            return []
    
    # =========================================================================
    # Channel Summaries (Sync)
    # =========================================================================
    
    def save_channel_summary(self, channel_id: int, summary_text: str) -> None:
        """
        Save L2 summary for a channel (sync).
        
        Args:
            channel_id: Discord channel ID
            summary_text: Summary text to save
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO channel_summaries 
                    (channel_id, summary_text, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (channel_id, summary_text))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save summary for channel {channel_id}: {e}")
    
    def load_channel_summaries(self) -> Dict[int, str]:
        """
        Load all channel summaries (sync).
        
        Returns:
            Dict mapping channel_id -> summary_text
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT channel_id, summary_text FROM channel_summaries"
                )
                return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Failed to load summaries: {e}")
            return {}
    
    def delete_channel_summary(self, channel_id: int) -> None:
        """
        Delete L2 summary for a channel (sync).
        
        Args:
            channel_id: Discord channel ID
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM channel_summaries WHERE channel_id = ?",
                    (channel_id,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete summary for channel {channel_id}: {e}")
    
    def get_stale_channels(self, hours: int = 24) -> List[int]:
        """
        Get channel IDs that haven't been updated in N hours (for cleanup).
        
        Args:
            hours: Number of hours of inactivity before channel is considered stale
            
        Returns:
            List of stale channel IDs
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT channel_id FROM channel_summaries 
                    WHERE last_updated < datetime('now', ? || ' hours')
                """, (f"-{hours}",))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get stale channels: {e}")
            return []
            return []

    # =========================================================================
    # Chat Logs (Sync)
    # =========================================================================

    def log_chat_message(
        self,
        guild_id: Optional[int],
        channel_id: int,
        user_id: int,
        user_name: str,
        content: str,
        is_bot: bool,
        has_attachments: bool
    ) -> None:
        """
        Log a chat message to the database (sync).
        
        Args:
            guild_id: Discord guild ID (can be None for DM)
            channel_id: Discord channel ID
            user_id: Discord user ID
            user_name: Username (for readability)
            content: Message content
            is_bot: Whether the message is from a bot
            has_attachments: Whether the message has attachments
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_logs 
                    (guild_id, channel_id, user_id, user_name, content, is_bot, has_attachments)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (guild_id, channel_id, user_id, user_name, content, int(is_bot), int(has_attachments)))
                conn.commit()
        except Exception as e:
            logger.error(f"event=log_chat_message_failed error={str(e)}")
    
    # =========================================================================
    # Channel Policies (Sync)
    # =========================================================================
    
    def save_channel_policy(
        self,
        channel_id: int,
        response_mode: str = "balanced",
        mood_adjustment: bool = True
    ) -> None:
        """
        Save channel-specific policy settings.
        
        Args:
            channel_id: Discord channel ID
            response_mode: Response mode ('strict', 'balanced', or 'chatty')
            mood_adjustment: Whether to enable mood-based adjustments
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO channel_policies 
                    (channel_id, response_mode, mood_adjustment, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (channel_id, response_mode, int(mood_adjustment)))
                conn.commit()
                logger.debug(
                    f"event=channel_policy_saved "
                    f"channel_id={channel_id} "
                    f"response_mode={response_mode}"
                )
        except Exception as e:
            logger.error(f"event=save_channel_policy_failed error={str(e)}")
    
    def get_channel_policy(self, channel_id: int) -> Optional[Tuple[str, bool]]:
        """
        Get channel-specific policy settings.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            Tuple of (response_mode, mood_adjustment) or None if not set
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT response_mode, mood_adjustment 
                    FROM channel_policies 
                    WHERE channel_id = ?
                """, (channel_id,))
                result = cursor.fetchone()
                if result:
                    return (result[0], bool(result[1]))
                return None
        except Exception as e:
            logger.error(f"event=get_channel_policy_failed error={str(e)}")
            return None
    
    def delete_channel_policy(self, channel_id: int) -> None:
        """
        Delete channel policy settings.
        
        Args:
            channel_id: Discord channel ID
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM channel_policies WHERE channel_id = ?",
                    (channel_id,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"event=delete_channel_policy_failed error={str(e)}")
    
    # =========================================================================
    # Async Wrappers (for non-blocking operations in event loop)
    # =========================================================================
    
    async def async_save_active_channel(
        self, 
        guild_id: int, 
        channel_id: int, 
        channel_name: str
    ) -> None:
        """Async wrapper for save_active_channel."""
        await asyncio.to_thread(self.save_active_channel, guild_id, channel_id, channel_name)
    
    async def async_remove_active_channel(self, guild_id: int) -> None:
        """Async wrapper for remove_active_channel."""
        await asyncio.to_thread(self.remove_active_channel, guild_id)
    
    async def async_load_active_channels(self) -> List[Tuple[int, int, str]]:
        """Async wrapper for load_active_channels."""
        return await asyncio.to_thread(self.load_active_channels)
    
    async def async_save_channel_summary(self, channel_id: int, summary_text: str) -> None:
        """Async wrapper for save_channel_summary."""
        await asyncio.to_thread(self.save_channel_summary, channel_id, summary_text)
    
    async def async_load_channel_summaries(self) -> Dict[int, str]:
        """Async wrapper for load_channel_summaries."""
        return await asyncio.to_thread(self.load_channel_summaries)
    
    async def async_delete_channel_summary(self, channel_id: int) -> None:
        """Async wrapper for delete_channel_summary."""
        await asyncio.to_thread(self.delete_channel_summary, channel_id)
    
    async def async_get_stale_channels(self, hours: int = 24) -> List[int]:
        """Async wrapper for get_stale_channels."""
        return await asyncio.to_thread(self.get_stale_channels, hours)

    async def async_log_chat_message(
        self,
        guild_id: Optional[int],
        channel_id: int,
        user_id: int,
        user_name: str,
        content: str,
        is_bot: bool,
        has_attachments: bool
    ) -> None:
        """Async wrapper for log_chat_message."""
        await asyncio.to_thread(
            self.log_chat_message, 
            guild_id, channel_id, user_id, user_name, content, is_bot, has_attachments
        )
    
    async def async_save_channel_policy(
        self,
        channel_id: int,
        response_mode: str = "balanced",
        mood_adjustment: bool = True
    ) -> None:
        """Async wrapper for save_channel_policy."""
        await asyncio.to_thread(
            self.save_channel_policy, channel_id, response_mode, mood_adjustment
        )
    
    async def async_get_channel_policy(self, channel_id: int) -> Optional[Tuple[str, bool]]:
        """Async wrapper for get_channel_policy."""
        return await asyncio.to_thread(self.get_channel_policy, channel_id)
    
    async def async_delete_channel_policy(self, channel_id: int) -> None:
        """Async wrapper for delete_channel_policy."""
        await asyncio.to_thread(self.delete_channel_policy, channel_id)
