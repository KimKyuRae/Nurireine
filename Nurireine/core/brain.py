
import logging
import time
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional, Tuple

from ..ai import Gatekeeper, MemoryManager, MainLLM
from ..database import DatabaseManager
from .. import config
from .types import UserInfo

logger = logging.getLogger(__name__)

class Brain:
    """
    Core AI logic engine for Nurireine.
    Decoupled from Discord for testing and multi-platform support.
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.gatekeeper: Optional[Gatekeeper] = None
        self.memory: Optional[MemoryManager] = None
        self.llm: Optional[MainLLM] = None
        self.is_loaded = False
        
        # Rate Limiting State
        self._llm_call_timestamps: list[float] = []
        
    def load_components(self):
        """Initialize AI components (blocking)."""
        logger.info("Initializing Brain components...")
        self.gatekeeper = Gatekeeper()
        self.memory = MemoryManager(self.gatekeeper, self.db)
        self.llm = MainLLM()
        self.is_loaded = True
        logger.info("Brain components loaded.")
        
    async def analyze(
        self,
        channel_id: int,
        content: str,
        user_info: UserInfo,
        guild_id: Optional[str] = None,
        is_explicit: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run SLM analysis via memory manager.
        """
        if not self.is_loaded:
            return {
                "l3_facts": "",
                "l2_summary": "AI 시스템이 아직 로드되지 않았습니다.",
                "l1_recent": []
            }, {}
            
        loop = asyncio.get_running_loop()
        
        try:
            # Use lambda to pass all arguments
            context, analysis = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.memory.plan_response(
                        channel_id, 
                        content, 
                        guild_id, 
                        user_info.id, 
                        user_info.name, 
                        is_explicit=is_explicit
                    )
                ),
                timeout=config.bot.response_timeout
            )
            return context, analysis
            
        except Exception as e:
            logger.error(f"Brain Analysis Error: {repr(e)}", exc_info=True)
            return {
                "l3_facts": "",
                "l2_summary": "오류가 발생했습니다.",
                "l1_recent": []
            }, {"response_needed": True}

    async def generate_response_stream(
        self,
        channel_id: int,
        user_input: str,
        context: Dict[str, Any],
        user_info: UserInfo,
        bot_info: UserInfo
    ) -> AsyncGenerator[str, None]:
        """
        Generate LLM response stream.
        """
        if not self.llm:
            yield "(LLM not loaded)"
            return
            
        # Add metadata to context
        context['user_id'] = user_info.id
        context['user_name'] = user_info.name
        context['bot_id'] = bot_info.id
        context['guild_id'] = context.get('guild_id') # Prepared by caller or analyze
        
        async for chunk in self.llm.generate_response_stream(
            user_input, context,
            memory_manager=self.memory,
            channel_id=channel_id
        ):
            yield chunk

    def save_interaction(
        self,
        channel_id: int,
        user_input: str,
        model_output: str,
        user_info: UserInfo
    ):
        """Save the turn to memory."""
        if self.memory:
            self.memory.add_message(
                channel_id,
                "user",
                user_input,
                user_name=user_info.name,
                user_id=user_info.id
            )
            self.memory.add_message(channel_id, "assistant", model_output)

    def check_rate_limit(self, is_explicit: bool, analysis: Dict[str, Any]) -> bool:
        """
        Check if an LLM call is allowed under the rate limit.
        """
        now = time.time()
        _LLM_RATE_LIMIT = 1
        _LLM_RATE_WINDOW = 20.0
        _PRIORITY_THRESHOLD = 0.85

        # Purge expired timestamps
        self._llm_call_timestamps = [
            t for t in self._llm_call_timestamps 
            if now - t < _LLM_RATE_WINDOW
        ]
        
        if is_explicit:
            self._llm_call_timestamps.append(now)
            return True
            
        perf_stats = analysis.get("_perf_stats", {})
        bert_score = perf_stats.get("bert_score", 0.0)
        
        if bert_score >= _PRIORITY_THRESHOLD:
            self._llm_call_timestamps.append(now)
            return True
            
        if len(self._llm_call_timestamps) >= _LLM_RATE_LIMIT:
            return False
            
        self._llm_call_timestamps.append(now)
        return True
