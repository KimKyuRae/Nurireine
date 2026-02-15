"""
Debug WebSocket Server

Provides real-time event broadcasting for visualization and debugging.
"""

import asyncio
import json
import logging
from typing import Set, Optional

import websockets
from websockets.server import WebSocketServerProtocol

from . import config

logger = logging.getLogger(__name__)


class DebugServer:
    """
    WebSocket server for real-time debugging events.
    
    Singleton pattern ensures only one server instance exists.
    Broadcasts events to all connected visualization clients.
    """
    
    _instance: Optional["DebugServer"] = None
    
    def __new__(cls) -> "DebugServer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._clients: Set[WebSocketServerProtocol] = set()
            cls._instance._server = None
        return cls._instance
    
    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)
    
    async def _register_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new client connection."""
        self._clients.add(websocket)
        logger.info(f"Visualizer connected. Total clients: {self.client_count}")
        
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logger.info(f"Visualizer disconnected. Total clients: {self.client_count}")
    
    async def start(
        self, 
        host: str = None, 
        port: int = None
    ) -> None:
        """
        Start the WebSocket server.
        
        Args:
            host: Hostname to bind to
            port: Port to listen on
        """
        host = host or config.debug.websocket_host
        port = port or config.debug.websocket_port
        
        logger.info(f"Starting Debug WebSocket Server on ws://{host}:{port}")
        self._server = await websockets.serve(self._register_client, host, port)
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Debug WebSocket Server stopped.")
    
    async def broadcast(self, event_type: str, data: dict) -> None:
        """
        Broadcast an event to all connected clients.
        
        Args:
            event_type: Type of event (e.g., 'user_input', 'llm_generate')
            data: Event data dictionary
        """
        if not self._clients:
            return
        
        message = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        websockets.broadcast(self._clients, message)


# ==============================================================================
# Global Helpers
# ==============================================================================

_server = DebugServer()


def get_server() -> DebugServer:
    """Get the singleton debug server instance."""
    return _server


def broadcast_event(event_type: str, data: dict) -> None:
    """
    Schedule a broadcast event.
    
    Safe to call from any async context. Does nothing if no event loop.
    
    Args:
        event_type: Type of event
        data: Event data dictionary
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_server.broadcast(event_type, data))
    except RuntimeError:
        pass  # No running event loop
