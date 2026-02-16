"""
LLM Service Module

Unified service for managing LLM clients (Gemini, G4F), handling API key rotation,
quota management, and providing consistent async interfaces.
"""

import logging
import random
import asyncio
from typing import List, Optional, Any, AsyncGenerator

from google import genai
from google.genai import types

try:
    from g4f.client import Client as G4FClient, AsyncClient as G4FAsyncClient
    from g4f.Provider import ApiAirforce
    G4F_AVAILABLE = True
except ImportError:
    G4F_AVAILABLE = False

from .. import config

logger = logging.getLogger(__name__)

class LLMService:
    """
    Centralized service for LLM interactions.
    Handles client initialization, key rotation, and error handling.
    """
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._init_gemini_clients()
        self._init_fallback()
        self._initialized = True
        
        logger.info(
            f"LLMService initialized. "
            f"Primary: Gemini ({len(self._gemini_clients)} keys), "
            f"Fallback: {'G4F' if self._fallback_client else 'Inactive'}"
        )

    def _init_gemini_clients(self) -> None:
        """Initialize Gemini clients for each API key."""
        self._gemini_clients: List[genai.Client] = []
        self._gemini_model_id = config.llm.model_id
        self._current_key_index = 0
        
        api_keys = config.llm.api_keys
        
        if not api_keys:
            logger.error("No GEMINI_API_KEY(s) set in environment variables.")
            return
        
        for key in api_keys:
            try:
                # Initialize client with http_options version for consistency
                client = genai.Client(api_key=key, http_options={'api_version': 'v1beta'})
                self._gemini_clients.append(client)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client with key ...{key[-4:]}: {e}")
                
        if self._gemini_clients:
            logger.info(f"Initialized {len(self._gemini_clients)} Gemini clients.")
        else:
            logger.error("Failed to initialize any Gemini clients.")

    def _init_fallback(self) -> None:
        """Initialize G4F fallback client."""
        self._fallback_client = None
        self._fallback_model_id = config.llm.fallback_model_id
        
        if not G4F_AVAILABLE:
            logger.warning("g4f library not found. Fallback disabled. (pip install g4f)")
            return
        
        try:
            # Use AsyncClient with specific provider
            self._fallback_client = G4FAsyncClient(provider=ApiAirforce)
            logger.info(f"Fallback LLM (G4F) initialized: {self._fallback_model_id}")
        except Exception as e:
            # Try basic client if Async fails
            try:
                 self._fallback_client = G4FClient(provider=ApiAirforce)
                 logger.warning(f"Fallback LLM (G4F) initialized in SYNC mode (Async failed): {e}")
            except Exception as e2:
                logger.error(f"Failed to initialize G4F client: {e2}")

    def _get_next_gemini_client(self) -> Optional[genai.Client]:
        """Get the next Gemini client in rotation."""
        if not self._gemini_clients:
            return None
            
        # Round-robin selection
        client = self._gemini_clients[self._current_key_index]
        self._current_key_index = (self._current_key_index + 1) % len(self._gemini_clients)
        return client
    
    def get_random_gemini_client(self) -> Optional[genai.Client]:
        """Get a random Gemini client (good for stateless requests like Gatekeeper)."""
        if not self._gemini_clients:
            return None
        return random.choice(self._gemini_clients)

    async def generate_content_async(
        self,
        contents: Any,
        model: str = None,
        config: Optional[types.GenerateContentConfig] = None
    ) -> Optional[types.GenerateContentResponse]:
        """
        Generate content using Gemini asynchronously with retry logic.
        """
        if not self._gemini_clients:
            logger.error("No Gemini clients available for generation.")
            return None

        model_id = model or self._gemini_model_id
        max_retries = len(self._gemini_clients)
        
        for attempt in range(max_retries):
            client = self._get_next_gemini_client()
            if not client:
                continue
                
            try:
                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                is_quota_error = "429" in str(e) or "ResourceExhausted" in str(type(e).__name__)
                
                if is_quota_error:
                    logger.warning(f"Gemini quota exceeded (Attempt {attempt+1}/{max_retries}). Rotating key...")
                    continue
                else:
                    logger.error(f"Gemini API failed: {type(e).__name__}: {e}")
                    # For non-quota errors, maybe we shouldn't retry immediately?
                    # But for now let's continue to try other keys just in case it's key-specific?
                    # Actually usually non-quota errors are bad request or server error.
                    continue
        
        return None

    async def generate_content_stream_async(
        self,
        contents: Any,
        model: str = None,
        config: Optional[types.GenerateContentConfig] = None
    ) -> AsyncGenerator[Any, None]:
        """
        Stream content using Gemini asynchronously with retry logic.
        """
        if not self._gemini_clients:
            logger.error("No Gemini clients available for streaming.")
            return

        model_id = model or self._gemini_model_id
        max_retries = len(self._gemini_clients)
        
        for attempt in range(max_retries):
            client = self._get_next_gemini_client()
            if not client:
                continue
            
            try:
                response_stream = await client.aio.models.generate_content_stream(
                    model=model_id,
                    contents=contents,
                    config=config
                )
                
                # Check if the stream is valid by yielding the first chunk?
                # generate_content_stream returns an async generator or similar wrapper
                async for chunk in response_stream:
                    yield chunk
                
                # If we successfully iterated, return
                return

            except Exception as e:
                is_quota_error = "429" in str(e) or "ResourceExhausted" in str(type(e).__name__)
                
                if is_quota_error:
                    logger.warning(f"Gemini quota exceeded during stream setup (Attempt {attempt+1}/{max_retries}). Rotating key...")
                    continue
                else:
                    logger.error(f"Gemini stream failed: {type(e).__name__}: {e}")
                    continue

    @property
    def fallback_client(self):
        return self._fallback_client

    @property
    def fallback_model_id(self):
        return self._fallback_model_id
