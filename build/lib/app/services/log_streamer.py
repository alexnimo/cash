"""
Real-time log streaming service for sending backend logs to frontend console.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Set
from fastapi import WebSocket, WebSocketDisconnect
from app.core.settings import get_settings

settings = get_settings()

class WebSocketLogHandler(logging.Handler):
    """Custom log handler that streams logs to connected WebSocket clients."""
    
    def __init__(self):
        super().__init__()
        self.connections: Set[WebSocket] = set()
        
    def add_connection(self, websocket: WebSocket):
        """Add a WebSocket connection to receive logs."""
        self.connections.add(websocket)
        
    def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.connections.discard(websocket)
        
    def emit(self, record: logging.LogRecord):
        """Send log record to all connected WebSocket clients."""
        if not self.connections:
            return
            
        try:
            # Format the log message
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname.lower(),
                "message": self.format(record),
                "module": record.name
            }
            
            # Send to all connected clients (fire and forget)
            asyncio.create_task(self._broadcast_log(log_data))
            
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Error in WebSocket log handler: {e}")
    
    async def _broadcast_log(self, log_data: dict):
        """Broadcast log data to all connected WebSocket clients."""
        if not self.connections:
            return
            
        message = json.dumps({
            "type": "log",
            "data": log_data
        })
        
        # Send to all connections, removing disconnected ones
        disconnected = set()
        for websocket in self.connections.copy():
            try:
                await websocket.send_text(message)
            except Exception:
                # Connection is likely closed
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.connections -= disconnected

# Global log streamer instance
_log_streamer = None

def get_log_streamer() -> WebSocketLogHandler:
    """Get the global log streamer instance."""
    global _log_streamer
    if _log_streamer is None:
        _log_streamer = WebSocketLogHandler()
        
        # Set up log formatting
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
        _log_streamer.setFormatter(formatter)
        
        # Add to relevant loggers
        _setup_log_streaming()
    
    return _log_streamer

def _setup_log_streaming():
    """Set up log streaming for relevant loggers."""
    global _log_streamer
    if not _log_streamer:
        return
        
    # Add to main application loggers
    loggers_to_stream = [
        'app.main',
        'app.services.video_processor',
        'app.services.content_analyzer', 
        'app.services.video_queue',
        'app.services.janitor_service',
        'uvicorn.access'
    ]
    
    for logger_name in loggers_to_stream:
        logger = logging.getLogger(logger_name)
        logger.addHandler(_log_streamer)
        # Ensure the logger level allows info and above
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

async def websocket_log_endpoint(websocket: WebSocket):
    """WebSocket endpoint for log streaming."""
    await websocket.accept()
    
    log_streamer = get_log_streamer()
    log_streamer.add_connection(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "message": "Log streaming connected",
                "level": "info"
            }
        }))
        
        # Keep connection alive
        while True:
            # Wait for any message from client (like ping/pong)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "data": {"timestamp": datetime.now().isoformat()}
                }))
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket log streaming error: {e}")
    finally:
        log_streamer.remove_connection(websocket)

def stream_log(message: str, level: str = "info", module: str = "app"):
    """Convenience function to stream a log message."""
    logger = logging.getLogger(module)
    
    level_map = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical
    }
    
    log_func = level_map.get(level.lower(), logger.info)
    log_func(message)
