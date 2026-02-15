"""
Nurireine - AI Discord Chatbot

Entry point for running the bot.
"""

import os
import sys
import asyncio
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from Nurireine import Nurireine, __version__
from Nurireine import config

# Configure logging
log_level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    logger.info(f"Starting Nurireine v{__version__} (Debug: {config.DEBUG_MODE})")
    
    # Validate configuration
    try:
        config.validate_all()
    except ValueError as e:
        logger.critical(f"Configuration Error: {e}")
        return 1
    
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.critical("DISCORD_TOKEN is missing from environment variables!")
        return 1
    
    try:
        # Fix for Windows event loop closed error
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        bot = Nurireine()
        bot.run(token)
        return 0
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())