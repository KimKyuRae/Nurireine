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

# Skip validation in test mode
import os
_IN_TEST_MODE = (
    os.getenv("PYTEST_CURRENT_TEST") is not None or 
    os.getenv("SKIP_CONFIG_VALIDATION") == "true"
)

# Validate configuration at startup only if not in test mode
if not _IN_TEST_MODE:
    from . import config
    try:
        config.validate_all()
    except ValueError as e:
        import logging
        logging.getLogger(__name__).error(f"Configuration validation failed: {e}")
        raise

# Only import bot if not in test mode (to avoid Discord dependency)
if not _IN_TEST_MODE:
    from .bot import Nurireine
    __all__ = ["Nurireine", "__version__"]
else:
    __all__ = ["__version__"]
