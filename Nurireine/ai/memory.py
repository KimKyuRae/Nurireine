"""
Memory Manager Module

Manages the three-layer memory system:
- L1: Short-term buffer (per channel, raw messages)
- L2: Rolling summary (per channel, structured markdown)
- L3: Long-term vector DB (ChromaDB, persistent facts)
"""

import logging
import time
import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import chromadb

from .. import config
from ..database import DatabaseManager
from ..debug_server import broadcast_event
from .embeddings import GGUFEmbeddingFunction
from .gatekeeper import Gatekeeper

logger = logging.getLogger(__name__)


# =============================================================================
# L2 Summary Data Structure (Markdown-based)
# =============================================================================

@dataclass
class L2Summary:
    """
    Structured L2 summary stored in markdown format.
    
    The markdown structure allows for organized, hierarchical information
    that's both human-readable and easy for LLMs to parse.
    """
    
    # Conversation overview
    topic: str = ""
    mood: str = ""
    
    # Participants
    users: List[str] = field(default_factory=list)
    
    # Context
    ongoing_topics: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    
    # Permanent Core (Immutable)
    permanent_core: List[str] = field(default_factory=list)
    
    # Conversation state
    last_speaker: str = ""
    conversation_stage: str = "시작"  # 시작, 진행중, 마무리
    
    # Raw for backward compatibility
    raw_text: str = ""
    
    @classmethod
    def from_markdown(cls, md_text: str) -> "L2Summary":
        """Parse markdown text into L2Summary structure."""
        summary = cls(raw_text=md_text)
        
        if not md_text:
            return summary
        
        # Parse sections using regex
        current_section = None
        lines = md_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for headers
            if line.startswith('## '):
                current_section = line[3:].strip().lower()
                continue
            elif line.startswith('# '):
                # Title/topic
                summary.topic = line[2:].strip()
                continue
            
            # Parse content based on section
            if not line or line.startswith('---'):
                continue
            
            if current_section == '분위기' or current_section == 'mood':
                summary.mood = line.lstrip('- ').strip()
            elif current_section == '참여자' or current_section == 'participants':
                if line.startswith('- '):
                    summary.users.append(line[2:].strip())
            elif current_section == '진행 중인 주제' or current_section == 'ongoing topics':
                if line.startswith('- '):
                    summary.ongoing_topics.append(line[2:].strip())
            elif current_section == '핵심 포인트' or current_section == 'key points':
                if line.startswith('- '):
                    summary.key_points.append(line[2:].strip())
            elif current_section == '대화 상태' or current_section == 'conversation state':
                if '마지막 발화자' in line or 'last speaker' in line.lower():
                    summary.last_speaker = line.split(':')[-1].strip()
                elif '단계' in line or 'stage' in line.lower():
                    summary.conversation_stage = line.split(':')[-1].strip()
            elif current_section == '불변 핵심' or current_section == 'permanent core':
                 if line.startswith('- '):
                    summary.permanent_core.append(line[2:].strip())
        
        return summary
    
    def to_markdown(self) -> str:
        """Convert L2Summary to markdown format."""
        lines = []
        
        # Title/Topic
        if self.topic:
            lines.append(f"# {self.topic}")
            lines.append("")
        
        # Mood
        if self.mood:
            lines.append("## 분위기")
            lines.append(f"- {self.mood}")
            lines.append("")
        
        # Participants
        if self.users:
            lines.append("## 참여자")
            for user in self.users:
                lines.append(f"- {user}")
            lines.append("")
        
        # Ongoing topics
        if self.ongoing_topics:
            lines.append("## 진행 중인 주제")
            for topic in self.ongoing_topics:
                lines.append(f"- {topic}")
            lines.append("")
        
        # Key points
        if self.key_points:
            lines.append("## 핵심 포인트")
            for point in self.key_points:
                lines.append(f"- {point}")
            lines.append("")
        
            lines.append("")
        
        # Permanent Core
        if self.permanent_core:
            lines.append("## 불변 핵심")
            for core in self.permanent_core:
                lines.append(f"- {core}")
            lines.append("")
        
        # Conversation state
        lines.append("## 대화 상태")
        lines.append(f"- 단계: {self.conversation_stage}")
        if self.last_speaker:
            lines.append(f"- 마지막 발화자: {self.last_speaker}")
        
        return "\n".join(lines)
    
    def to_context_string(self) -> str:
        """
        Convert to a concise string for LLM context.
        More compact than full markdown for token efficiency.
        """
        parts = []
        
        if self.topic:
            parts.append(f"[주제] {self.topic}")
        
        if self.mood:
            parts.append(f"[분위기] {self.mood}")
        
        if self.ongoing_topics:
            topics_str = ", ".join(self.ongoing_topics[:3])
            parts.append(f"[진행 중] {topics_str}")
        
        if self.key_points:
            points_str = "; ".join(self.key_points[:5])
            parts.append(f"[핵심] {points_str}")
        
        if self.permanent_core:
            core_str = "; ".join(self.permanent_core)
            parts.append(f"[불변] {core_str}")
        
        parts.append(f"[상태] {self.conversation_stage}")
        
        return "\n".join(parts)
    
    @classmethod
    def create_initial(cls) -> "L2Summary":
        """Create an initial empty summary."""
        return cls(
            topic="새로운 대화",
            mood="중립",
            conversation_stage="시작"
        )


# =============================================================================
# Default L2 Template
# =============================================================================

DEFAULT_L2_TEMPLATE = """# 새로운 대화

## 분위기
- 중립

## 참여자

## 진행 중인 주제

## 핵심 포인트

## 불변 핵심

## 대화 상태
- 단계: 시작
"""


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryManager:
    """
    Manages the layered memory system for the bot.
    
    Memory Layers:
    - L1 (Buffer): Recent messages per channel, stored in memory
    - L2 (Summary): Rolling markdown summary per channel, persisted to DB
    - L3 (Facts): Long-term facts in ChromaDB vector database
    
    Thread Safety:
    - Uses threading.Lock for L1 buffer modifications
    - Uses threading.Lock for L2 summary modifications
    """
    
    # Maximum number of L2 summaries to keep in memory (LRU-style cleanup)
    MAX_L2_CACHE_SIZE = 100
    
    def __init__(
        self, 
        gatekeeper: Gatekeeper,
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            gatekeeper: Gatekeeper instance for SLM analysis
            db_manager: Database manager for L2 persistence
        """
        import threading
        
        self.gatekeeper = gatekeeper
        self.db = db_manager
        
        # Async locks for buffer/summary access
        self._l1_lock = asyncio.Lock()
        self._l2_lock = asyncio.Lock()
        
        # L1: Channel ID -> List of messages
        self._l1_buffers: Dict[int, List[Dict[str, str]]] = {}
        
        # L2: Channel ID -> L2Summary object with access timestamps
        self._l2_summaries: Dict[int, L2Summary] = {}
        self._l2_access_times: Dict[int, float] = {}  # Track last access for LRU cleanup
        self._load_initial_summaries()
        
        # L3: ChromaDB collection
        self._chroma_healthy = False
        self._init_vector_db()
        
        # Dirty counter for state-aware analysis (offer.md §2-가)
        self._dirty_counters: Dict[int, int] = {}  # channel_id -> message count since last analysis
        self._last_analysis_time: Dict[int, float] = {}  # channel_id -> timestamp
        self._last_analysis_result: Dict[int, Dict[str, Any]] = {}  # channel_id -> last analysis
        
        # Initialize base lore
        self._initialize_lore()
    
    async def _load_summaries_from_db(self) -> None:
        """Deprecated: Use _load_initial_summaries instead."""
        pass

    def _load_initial_summaries(self) -> None:
        """Load initial L2 summaries from database (Sync, for init)."""
        if not self.db:
            return
        
        # We are in a worker thread (from bot.py), so blocking IO is fine.
        try:
            loaded = self.db.load_channel_summaries()
            
            if loaded:
                current_time = time.time()
                # No lock needed during initialization
                for channel_id, md_text in loaded.items():
                    self._l2_summaries[channel_id] = L2Summary.from_markdown(md_text)
                    self._l2_access_times[channel_id] = current_time
                logger.info(f"Loaded L2 summaries for {len(loaded)} channels from DB.")
        except Exception as e:
            logger.error(f"Failed to load initial summaries: {e}")
    
    def _init_vector_db(self) -> None:
        """Initialize ChromaDB with custom embedding function."""
        logger.info("Initializing ChromaDB with GGUF embedding function...")
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(config.CHROMA_DB_DIR)
            )
            self._embedding_fn = GGUFEmbeddingFunction()
            
            self._collection = self._chroma_client.get_or_create_collection(
                name=config.memory.collection_name,
                embedding_function=self._embedding_fn
            )
            self._chroma_healthy = True
            logger.info("ChromaDB ready.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._chroma_healthy = False
            raise
    
    def check_chroma_health(self) -> bool:
        """
        Check if ChromaDB connection is healthy.
        Attempts reconnection if unhealthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._chroma_healthy:
            try:
                logger.info("Attempting ChromaDB reconnection...")
                self._init_vector_db()
            except Exception as e:
                logger.error(f"ChromaDB reconnection failed: {e}")
                return False
        
        # Health check: try a simple operation
        try:
            self._collection.count()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            self._chroma_healthy = False
            return False
    
    def _initialize_lore(self) -> None:
        """Initialize base character lore in L3 memory."""
        if not self._chroma_healthy:
            logger.warning("ChromaDB unhealthy, skipping lore initialization.")
            return
            
        if self._collection.count() > 0:
            return
        
        logger.info("Initializing base lore into L3 memory...")
        self.save_facts(config.BASE_LORE, context_type="lore")
        logger.info("Base lore initialized.")
    
    async def cleanup_stale_l2(self, max_age_hours: int = 24) -> int:
        """
        Remove L2 summaries that haven't been accessed recently (Async).
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned = 0
        loop = asyncio.get_running_loop()
        
        async with self._l2_lock:
            stale_channels = [
                channel_id for channel_id, access_time in self._l2_access_times.items()
                if current_time - access_time > max_age_seconds
            ]
            
            for channel_id in stale_channels:
                # Save to DB before removing
                if self.db and channel_id in self._l2_summaries:
                    md_text = self._l2_summaries[channel_id].to_markdown()
                    await loop.run_in_executor(None, self.db.save_channel_summary, channel_id, md_text)
                
                # Remove from memory
                self._l2_summaries.pop(channel_id, None)
                self._l2_access_times.pop(channel_id, None)
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale L2 summaries.")
        
        return cleaned
    
    def _enforce_l2_cache_limit(self) -> None:
        """Remove oldest L2 summaries if cache exceeds limit (Deprecated sync call, logic moved to async methods)."""
        pass
        
    async def _enforce_l2_cache_limit_async(self) -> None:
        """Remove oldest L2 summaries if cache exceeds limit (LRU eviction)."""
        loop = asyncio.get_running_loop()
        # Lock is held by caller
        if len(self._l2_summaries) > self.MAX_L2_CACHE_SIZE:
            sorted_channels = sorted(
                self._l2_access_times.items(),
                key=lambda x: x[1]
            )
            
            to_remove = len(self._l2_summaries) - self.MAX_L2_CACHE_SIZE
            for channel_id, _ in sorted_channels[:to_remove]:
                if self.db and channel_id in self._l2_summaries:
                    md_text = self._l2_summaries[channel_id].to_markdown()
                    await loop.run_in_executor(None, self.db.save_channel_summary, channel_id, md_text)
                
                self._l2_summaries.pop(channel_id, None)
                self._l2_access_times.pop(channel_id, None)
            
            logger.info(f"LRU evicted {to_remove} L2 summaries from cache.")
    
    # =========================================================================
    # L1 Buffer Operations
    # =========================================================================
    
    async def get_l1_buffer(self, channel_id: int) -> List[Dict[str, str]]:
        """Get the L1 buffer for a channel (Async)."""
        async with self._l1_lock:
            if channel_id not in self._l1_buffers:
                self._l1_buffers[channel_id] = []
            return list(self._l1_buffers[channel_id]) # Return copy to check concurrency
    
    async def add_message(
        self, 
        channel_id: int, 
        role: str, 
        content: str, 
        user_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Add a message to the L1 buffer (Async).
        
        Args:
            channel_id: Discord channel ID
            role: Message role ("user" or "assistant")
            content: Message content
            user_name: Optional user display name
            user_id: Optional user ID
        """
        from ..validation import input_validator
        
        # Validate inputs
        valid, error = input_validator.validate_channel_id(channel_id)
        if not valid:
            logger.error(f"Invalid channel_id: {error}")
            return
        
        if role not in ["user", "assistant"]:
            logger.error(f"Invalid role: {role}")
            return
        
        valid, error = input_validator.validate_message_content(content)
        if not valid:
            logger.warning(f"Invalid message content: {error}")
            # Sanitize rather than reject
            content = input_validator.sanitize_for_embedding(content)
        
        if user_name:
            valid, error = input_validator.validate_username(user_name)
            if not valid:
                logger.warning(f"Invalid username: {error}")
                user_name = user_name[:100]  # Truncate
        
        if user_id:
            valid, error = input_validator.validate_user_id(user_id)
            if not valid:
                logger.warning(f"Invalid user_id: {error}")
                user_id = None  # Drop invalid ID
        
        async with self._l1_lock:
            buffer = self._l1_buffers.setdefault(channel_id, [])
            msg_data = {"role": role, "content": content}
            if user_name:
                msg_data["user_name"] = user_name
            if user_id:
                msg_data["user_id"] = str(user_id)
                
            buffer.append(msg_data)
            
            if len(buffer) > config.memory.l1_buffer_limit:
                buffer.pop(0)
    
    async def clear_l1_buffer(self, channel_id: int) -> None:
        """Clear the L1 buffer for a channel (Async)."""
        async with self._l1_lock:
            if channel_id in self._l1_buffers:
                self._l1_buffers[channel_id].clear()
    
    # =========================================================================
    # L2 Summary Operations (Markdown-based)
    # =========================================================================
    
    async def get_l2_summary(self, channel_id: int) -> L2Summary:
        """Get the L2 summary for a channel (Async)."""
        async with self._l2_lock:
            if channel_id not in self._l2_summaries:
                self._l2_summaries[channel_id] = L2Summary.create_initial()
                await self._enforce_l2_cache_limit_async()
            
            self._l2_access_times[channel_id] = time.time()
            return self._l2_summaries[channel_id]
    
    async def get_l2_markdown(self, channel_id: int) -> str:
        """Get the L2 summary as markdown text (Async)."""
        summary = await self.get_l2_summary(channel_id)
        return summary.to_markdown()
    
    async def get_l2_context(self, channel_id: int) -> str:
        """Get the L2 summary as a concise context string (Async)."""
        summary = await self.get_l2_summary(channel_id)
        return summary.to_context_string()
    
    async def update_l2_from_markdown(self, channel_id: int, md_text: str) -> None:
        """
        Update the L2 summary from markdown text (Async).
        """
        if md_text:
            async with self._l2_lock:
                current_summary = self._l2_summaries.get(channel_id, L2Summary.create_initial())
                preserved_core = current_summary.permanent_core
                
                # Parse markdown (CPU bound but fast enough? Assuming yes)
                new_summary = L2Summary.from_markdown(md_text)
                
                if preserved_core:
                    new_summary.permanent_core = preserved_core
                    
                self._l2_summaries[channel_id] = new_summary
                self._l2_access_times[channel_id] = time.time()
    
    async def update_l2_fields(
        self, 
        channel_id: int,
        topic: Optional[str] = None,
        mood: Optional[str] = None,
        new_user: Optional[str] = None,
        new_topic: Optional[str] = None,
        new_point: Optional[str] = None,
        new_core: Optional[str] = None,
        last_speaker: Optional[str] = None,
        conversation_stage: Optional[str] = None
    ) -> None:
        """
        Update specific fields of the L2 summary (Async).
        """
        summary = await self.get_l2_summary(channel_id)
        
        # Note: get_l2_summary returns a reference, so modifications affect the object.
        # But we need to ensure thread safety if we are modifying it?
        # Since we are modifying fields in place, we should hold the lock during modification?
        # But get_l2_summary releases the lock.
        # Ideally, we should acquire lock, get summary, modify, release.
        
        async with self._l2_lock:
            # We already got the summary object, but let's re-fetch from cache to be safe or just use it.
            # Since self._l2_summaries holds the object, and we have a reference.
            # But another task might have replaced the object in the dictionary?
            # Yes, if update_l2_from_markdown ran.
            if channel_id in self._l2_summaries:
                summary = self._l2_summaries[channel_id]
            else:
                return # Should have been created by get_l2_summary call above if we wanted to be strict
            
            if topic:
                summary.topic = topic
            if mood:
                summary.mood = mood
            if new_user and new_user not in summary.users:
                summary.users.append(new_user)
            if new_topic and new_topic not in summary.ongoing_topics:
                # Filter out topics that are about the bot itself (meta-discussion)
                bot_aliases = ["누리야", "누리레느", "누리레인", "누리", "레느", "nurireine"]
                is_meta_topic = any(alias in new_topic.lower() for alias in bot_aliases)
                if not is_meta_topic:
                    summary.ongoing_topics.append(new_topic)
                else:
                    # Strip bot name and keep the actual topic if meaningful
                    cleaned = new_topic
                    for alias in bot_aliases:
                        cleaned = cleaned.replace(alias, "").strip()
                    # Only add if there's still meaningful content (e.g., "설정 논의")
                    if len(cleaned) >= 2 and cleaned not in summary.ongoing_topics:
                        summary.ongoing_topics.append(cleaned)
                # Keep only recent topics
                if len(summary.ongoing_topics) > 5:
                    summary.ongoing_topics.pop(0)
            if new_point:
                summary.key_points.append(new_point)
                # Keep only recent points
                if len(summary.key_points) > 10:
                    summary.key_points.pop(0)
            if new_core and new_core not in summary.permanent_core:
                summary.permanent_core.append(new_core)
            if last_speaker:
                summary.last_speaker = last_speaker
            if conversation_stage:
                summary.conversation_stage = conversation_stage
    
    async def save_all_summaries(self) -> None:
        """Persist all L2 summaries to database (Async)."""
        if not self.db:
            return
        
        loop = asyncio.get_running_loop()
        count = 0
        async with self._l2_lock:
            for channel_id, summary in self._l2_summaries.items():
                md_text = summary.to_markdown()
                await loop.run_in_executor(None, self.db.save_channel_summary, channel_id, md_text)
                count += 1
        logger.info(f"Saved L2 summaries for {count} channels to DB.")
    
    # =========================================================================
    # L3 Vector DB Operations
    # =========================================================================
    
    async def save_facts(
        self, 
        facts: List[Any], 
        context_type: str = "lore", 
        context_id: Optional[str] = None,
        guild_id: Optional[str] = None
    ) -> None:
        """
        Save facts to L3 vector database (Async wrapper).
        """
        if not facts:
            return
        
        loop = asyncio.get_running_loop()
        
        def _blocking_save():
            # Normalized input...
            normalized_facts = []
            for fact in facts:
                if isinstance(fact, str):
                    normalized_facts.append({"content": fact, "topic": "general", "keywords": []})
                elif isinstance(fact, dict) and "content" in fact:
                    normalized_facts.append(fact)
            
            # Filter duplicates...
            unique_facts = []
            for fact_obj in normalized_facts:
                fact_content = fact_obj["content"]
                try:
                    conditions = [{"context": context_type}]
                    if context_type == "guild" and context_id:
                        conditions.append({"guild_id": str(context_id)})
                    elif context_type == "user" and context_id:
                        conditions.append({"user_id": str(context_id)})
                    
                    where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]
                    
                    existing = self._collection.query(
                        query_texts=[fact_content],
                        n_results=1,
                        where=where_filter
                    )
                    
                    is_duplicate = False
                    if existing['distances'] and len(existing['distances'][0]) > 0:
                        distance = existing['distances'][0][0]
                        if distance < config.memory.l3_similarity_threshold:
                            is_duplicate = True
                    
                    if not is_duplicate:
                        unique_facts.append(fact_obj)
                except Exception as e:
                    unique_facts.append(fact_obj)
            
            if not unique_facts:
                return

            try:
                timestamp = int(time.time() * 1000)
                ids = [f"{context_type}_{timestamp}_{i}" for i in range(len(unique_facts))]
                
                documents = []
                metadatas = []
                
                for fact_obj in unique_facts:
                    content = fact_obj["content"]
                    topic = fact_obj.get("topic")
                    keywords = fact_obj.get("keywords", [])
                    if isinstance(keywords, list):
                        keywords_str = ",".join(str(k) for k in keywords)
                    else:
                        keywords_str = str(keywords)

                    meta = {
                        "context": context_type, 
                        "timestamp": time.time(),
                        "topic": str(topic) if topic else "general",
                        "keywords": keywords_str
                    }
                    if context_type == "guild" and context_id:
                        meta["guild_id"] = str(context_id)
                    elif context_type == "user" and context_id:
                        meta["user_id"] = str(context_id)
                        if guild_id:
                            meta["guild_id"] = str(guild_id)
                    
                    documents.append(content)
                    metadatas.append(meta)
                
                self._collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.error(f"Failed to save facts to L3: {e}")

        await loop.run_in_executor(None, _blocking_save)
    
    async def retrieve_facts(
        self, 
        query: str, 
        guild_id: Optional[str] = None, 
        user_id: Optional[str] = None
    ) -> str:
        """
        Retrieve relevant facts from L3 memory (Async wrapper).
        """
        if not query:
            return ""
        
        loop = asyncio.get_running_loop()
        
        def _blocking_retrieve():
            retrieved_docs = []
            try:
                lore_results = self._collection.query(
                    query_texts=[query],
                    n_results=2, 
                    where={"context": "lore"}
                )
                if lore_results['documents'] and lore_results['documents'][0]:
                     retrieved_docs.extend(lore_results['documents'][0])
            except Exception as e:
                pass

            conditions = []
            if guild_id:
                conditions.append({"guild_id": str(guild_id)})
            if user_id:
                conditions.append({"user_id": str(user_id)})
            
            if conditions:
                where_clause = {"$or": conditions} if len(conditions) > 1 else conditions[0]
                user_results = self._collection.query(
                    query_texts=[query],
                    n_results=4, # Prioritize user/guild facts
                    where=where_clause
                )
                if user_results['documents'] and user_results['documents'][0]:
                    retrieved_docs.extend(user_results['documents'][0])
            
            unique_docs = []
            seen = set()
            for doc in retrieved_docs:
                if doc not in seen:
                    unique_docs.append(doc)
                    seen.add(doc)
            
            if unique_docs:
                return "\n".join([f"- {text}" for text in unique_docs])
            return "관련된 기억이 없습니다."

        try:
            return await loop.run_in_executor(None, _blocking_retrieve)
        except Exception as e:
            logger.error(f"Fact retrieval failed: {e}")
            return ""
    
    # =========================================================================
    # Main Processing
    # =========================================================================
    
    async def plan_response(
        self, 
        channel_id: int, 
        user_input: str, 
        guild_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        is_explicit: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze user input and prepare context for LLM.
        
        This method:
        1. Updates L2 with new user info
        2. Checks if response is needed (BERT)
        3. Runs full SLM analysis if needed (update summarry, extract facts)
        4. Retrieves relevant L3 facts
        5. Assembles final context
        """
        broadcast_event("memory_access", {"stage": "plan_start", "channel": channel_id})
        logger.info(f"Memory planning response for channel {channel_id} (Guild: {guild_id}, User: {user_id})")
        
        stats = {
            "bert": 0.0,
            "slm": 0.0,
            "l3_search": 0.0,
            "l3_save": 0.0
        }
        
        l1_buffer = await self.get_l1_buffer(channel_id)
        l2_summary = await self.get_l2_summary(channel_id)
        
        # Track user participant (updates l2_summary in place)
        if user_name:
            await self.update_l2_fields(channel_id, new_user=user_name, last_speaker=user_name)
            # Explicitly re-fetch after updates to ensure SLM gets latest state
            l2_summary = await self.get_l2_summary(channel_id)
        
        # Dirty counter logic (offer.md §2-가: State-aware Analysis)
        self._dirty_counters[channel_id] = self._dirty_counters.get(channel_id, 0) + 1
        dirty_count = self._dirty_counters[channel_id]
        analysis_interval = config.memory.analysis_interval
        
        # === STAGE 1: Always run BERT check (fast, ~10ms) ===
        # This determines if the bot should respond at all.
        # Even when we skip full SLM analysis, we must check this.
        should_respond = True
        if not is_explicit and self.gatekeeper:
            recent_for_bert = l1_buffer[-config.memory.l1_context_limit:]
            bert_context_parts = []
            for msg in recent_for_bert:
                content = msg.get('content', '')
                if msg.get('role') == 'assistant':
                    content = "[BOT_RESPONSE]"
                bert_context_parts.append(content)
            bert_context = " [SEP] ".join(bert_context_parts)
            
            is_needed, score, latency = await self.gatekeeper.check_response_needed(bert_context, user_input)
            logger.info(f"BERT check in plan_response: Needed={is_needed} (Score={score:.4f})")
            stats["bert_score"] = float(score)
            should_respond = is_needed
            
            if not is_needed:
                # No response needed — return early with empty analysis
                analysis = {
                    "response_needed": False,
                    "retrieval_needed": False,
                    "search_query": None,
                    "guild_facts": [],
                    "user_facts": [],
                    "summary_updates": {"topic": None, "mood": None, "new_topic": None, "new_point": None, "stage": None},
                    "_perf_stats": stats
                }
                context = {
                    "l3_facts": "",
                    "l2_summary": await self.get_l2_context(channel_id),
                    "l2_full": await self.get_l2_markdown(channel_id),
                    "l1_recent": l1_buffer[-config.memory.l1_llm_context_limit:],
                }
                broadcast_event("memory_access", {"stage": "plan_end"})
                return context, analysis
        
        # === STAGE 2: Full SLM analysis (gated by dirty_counter) ===
        should_analyze = (
            should_respond and
            (
                dirty_count >= analysis_interval
                or is_explicit
                or channel_id not in self._last_analysis_result
            )
        )
        
        if should_analyze:
            # === PHASE 1: Fast decision (response/retrieval + search query) ===
            t0 = time.time()
            analysis = await self._run_analysis(user_input, l2_summary.to_markdown(), l1_buffer, is_explicit, user_id=user_id, user_name=user_name)
            stats["slm_phase1"] = time.time() - t0
            
            # Reset dirty counter
            self._dirty_counters[channel_id] = 0
            self._last_analysis_time[channel_id] = time.time()
            self._last_analysis_result[channel_id] = analysis
            
            # Retrieve L3 facts if needed (blocking - needed for context)
            should_retrieve = analysis.get("retrieval_needed") or is_explicit
            l3_context = ""
            if should_retrieve:
                t0 = time.time()
                # Use SLM-provided query if available, otherwise fall back to cleaned user input
                query = analysis.get("search_query")
                if not query or query.strip() == "":
                    from ..utils.text_cleaner import clean_query_text
                    # Simple fallback: just clean the input (remove mentions, reply headers, etc.)
                    query = clean_query_text(user_input)
                    if not query:
                        query = user_input
                    logger.info(f"SLM search_query was empty, using cleaned input: '{query}' (from: '{user_input}')")
                else:
                    logger.info(f"Using SLM-provided search_query: '{query}'")

                logger.info(f"Retrieving L3 facts for query: '{query}' (Guild: {guild_id}, User: {user_id})")
                l3_context = await self.retrieve_facts(query, guild_id, user_id)
                stats["l3_search"] = time.time() - t0
                broadcast_event("memory_access", {"stage": "retrieved", "query": query, "found": bool(l3_context)})
            
            # === PHASE 2: Lazy extraction (facts + summary) - Run in background ===
            # Capture consistent snapshot of conversation state for background task
            # This avoids race conditions if l1_buffer is modified while extraction runs
            l2_snapshot = l2_summary.to_markdown()
            l1_snapshot = list(l1_buffer)  # Create a copy of the list
            
            # Schedule extraction to run asynchronously after response starts
            extraction_task = asyncio.create_task(
                self._run_lazy_extraction(
                    channel_id, user_input, l2_snapshot, l1_snapshot,
                    user_id, user_name, guild_id
                )
            )
            # Add error handler to log exceptions with full traceback
            def _log_extraction_error(task: asyncio.Task) -> None:
                try:
                    # Will raise if the task failed
                    task.result()
                except Exception as e:
                    logger.error(f"Lazy extraction task failed: {e}", exc_info=True)
            
            extraction_task.add_done_callback(_log_extraction_error)
        else:
            # Skip full SLM analysis — BERT already confirmed response is needed
            logger.info(f"Skipping SLM analysis (dirty_count={dirty_count}/{analysis_interval}), BERT confirmed response needed")
            analysis = self._last_analysis_result.get(channel_id, {
                "response_needed": True,
                "retrieval_needed": False,
                "search_query": None,
                "guild_facts": [],
                "user_facts": [],
                "summary_updates": {"topic": None, "mood": None, "new_topic": None, "new_point": None, "stage": None}
            })
            l3_context = ""
        
        # Inject stats into analysis for retrieval by bot
        analysis["_perf_stats"] = stats
        
        # 5. Build final context
        # Include minimal L1 recent (3 messages) for conversational coherence.
        # Full history is available via the get_chat_history tool for deeper lookback.
        context = {
            "l3_facts": l3_context,
            "l2_summary": await self.get_l2_context(channel_id),  # Use concise format
            "l2_full": await self.get_l2_markdown(channel_id),    # Full markdown available
            "l1_recent": l1_buffer[-config.memory.l1_llm_context_limit:],
        }
        
        broadcast_event("memory_access", {"stage": "plan_end"})
        return context, analysis
    
    async def _run_analysis(
        self, 
        user_input: str, 
        l2_markdown: str, 
        l1_buffer: List[Dict[str, str]],
        is_explicit: bool = False,
        user_id: str = None,
        user_name: str = None
    ) -> Dict[str, Any]:
        """Run Gatekeeper analysis (Async)."""
        if not self.gatekeeper:
            logger.warning("Gatekeeper unavailable, using default analysis.")
            return {
                "response_needed": True,
                "retrieval_needed": False,
                "search_query": None,
                "guild_facts": [],
                "user_facts": [],
                "summary_updates": {
                    "topic": None,
                    "mood": None,
                    "new_topic": None,
                    "new_point": None,
                    "stage": None
                }
            }
        
        # Prepare messages
        recent_messages = l1_buffer[-config.memory.l1_context_limit:]
        
        return await self.gatekeeper.process_turn(
            user_input, l2_markdown, recent_messages, 
            is_explicit=is_explicit,
            user_id=user_id or "unknown",
            user_name=user_name or "unknown",
            skip_bert=True  # BERT already checked in plan_response
        )
    
    async def _run_lazy_extraction(
        self,
        channel_id: int,
        user_input: str,
        l2_markdown: str,
        l1_buffer: List[Dict[str, str]],
        user_id: Optional[str],
        user_name: Optional[str],
        guild_id: Optional[int]
    ) -> None:
        """
        Run Phase 2 extraction (facts + summary) in background.
        This is called asynchronously after the response starts.
        """
        try:
            logger.info(f"Starting lazy extraction for channel {channel_id}")
            t0 = time.time()
            
            if not self.gatekeeper:
                logger.warning("Gatekeeper unavailable for lazy extraction.")
                return
            
            # Prepare messages
            recent_messages = l1_buffer[-config.memory.l1_context_limit:]
            
            # Run Phase 2 extraction
            extraction = await self.gatekeeper.run_extraction(
                user_input, l2_markdown, recent_messages,
                user_id=user_id or "unknown",
                user_name=user_name or "unknown"
            )
            
            extraction_time = time.time() - t0
            logger.info(f"Lazy extraction completed in {extraction_time:.3f}s")
            
            # Update L2 summary from extraction
            updates = extraction.get("summary_updates", {})
            if updates and any(updates.values()):
                valid_updates = {k: v for k, v in updates.items() if v}
                if valid_updates:
                    await self.update_l2_fields(
                        channel_id,
                        topic=valid_updates.get("topic"),
                        mood=valid_updates.get("mood"),
                        new_topic=valid_updates.get("new_topic"),
                        new_point=valid_updates.get("new_point"),
                        conversation_stage=valid_updates.get("stage")
                    )
            
            # Save new facts
            t0 = time.time()
            if extraction.get("guild_facts") and guild_id:
                await self.save_facts(extraction["guild_facts"], context_type="guild", context_id=guild_id)
            if extraction.get("user_facts") and user_id:
                await self.save_facts(extraction["user_facts"], context_type="user", context_id=user_id, guild_id=guild_id)
            save_time = time.time() - t0
            logger.info(f"Facts saved in {save_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error in lazy extraction for channel {channel_id}: {e}", exc_info=True)
