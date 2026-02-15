"""
Nurireine - A Discord AI Chatbot with Layered Memory System

This package provides a Discord bot with:
- L1 Memory: Short-term conversation buffer (per channel)
- L2 Memory: Rolling summary (per channel)  
- L3 Memory: Long-term vector database (ChromaDB)

Architecture:
- core/: Message handling and context management
- ai/: AI components (Gatekeeper, LLM, Memory, Embeddings)
- utils/: Utility functions
- cogs/: Discord command extensions
"""

__version__ = "2.0.0"
__author__ = "SapoKR"

# Validate configuration at startup (#10)
from . import config
try:
    config.validate_all()
except ValueError as e:
    import logging
    logging.getLogger(__name__).error(f"Configuration validation failed: {e}")
    raise

from .bot import Nurireine

__all__ = ["Nurireine", "__version__"]
