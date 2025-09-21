"""
Event Streaming Service - Stream specific events to frontend
Only streams important processing events, not debug/secret data
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any
from enum import Enum

from fastapi import WebSocket

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events to stream to frontend"""
    VIDEO_QUEUED = "video_queued"
    TRANSCRIPTION_START = "transcription_start"
    TRANSCRIPTION_PROGRESS = "transcription_progress"
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    IMAGE_ANALYSIS_START = "image_analysis_start"
    IMAGE_ANALYSIS_PROGRESS = "image_analysis_progress"
    IMAGE_ANALYSIS_COMPLETE = "image_analysis_complete"
    CONTENT_ANALYSIS_START = "content_analysis_start"
    CONTENT_ANALYSIS_COMPLETE = "content_analysis_complete"
    VIDEO_PROCESSING_COMPLETE = "video_processing_complete"
    ERROR_OCCURRED = "error_occurred"

class EventData:
    """Represents an event to be streamed"""
    
    def __init__(self, event_type: EventType, video_id: str, message: str, 
                 progress: Optional[int] = None, details: Optional[Dict] = None):
        self.event_type = event_type
        self.video_id = video_id
        self.message = message
        self.progress = progress
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": "event",
            "event_type": self.event_type.value,
            "video_id": self.video_id,
            "message": self.message,
            "progress": self.progress,
            "details": self.details,
            "timestamp": self.timestamp
        }

class EventStreamer:
    """Streams specific processing events to connected WebSocket clients"""
    
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.event_queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the event streaming service"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._stream_events())
            logger.info("Event streaming service started")
    
    async def stop(self):
        """Stop the event streaming service"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Event streaming service stopped")
    
    def add_connection(self, websocket: WebSocket):
        """Add a WebSocket connection"""
        self.connections.add(websocket)
        logger.info(f"Event streaming connection added. Total: {len(self.connections)}")
    
    def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.connections.discard(websocket)
        logger.info(f"Event streaming connection removed. Total: {len(self.connections)}")
    
    async def emit_event(self, event: EventData):
        """Emit an event to all connected clients"""
        if not self._running:
            await self.start()
        
        await self.event_queue.put(event)
    
    async def _stream_events(self):
        """Main event streaming loop"""
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Convert to JSON
                event_data = json.dumps(event.to_dict())
                
                # Send to all connected clients
                disconnected = set()
                for websocket in self.connections.copy():
                    try:
                        await websocket.send_text(event_data)
                    except Exception as e:
                        logger.warning(f"Failed to send event to client: {e}")
                        disconnected.add(websocket)
                
                # Remove disconnected clients
                for websocket in disconnected:
                    self.remove_connection(websocket)
                    
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in event streaming loop: {e}")
                await asyncio.sleep(1)

# Global event streamer instance
_event_streamer: Optional[EventStreamer] = None

def get_event_streamer() -> EventStreamer:
    """Get the global event streamer instance"""
    global _event_streamer
    if _event_streamer is None:
        _event_streamer = EventStreamer()
    return _event_streamer

def emit_video_event(event_type: EventType, video_id: str, message: str, 
                    progress: Optional[int] = None, details: Optional[Dict] = None):
    """
    Convenience function to emit a video processing event
    Can be called from anywhere in the application
    """
    streamer = get_event_streamer()
    event = EventData(event_type, video_id, message, progress, details)
    # Use asyncio.create_task to emit without awaiting
    asyncio.create_task(streamer.emit_event(event))

# WebSocket endpoint for event streaming
async def websocket_event_endpoint(websocket: WebSocket):
    """WebSocket endpoint for event streaming"""
    await websocket.accept()
    
    event_streamer = get_event_streamer()
    event_streamer.add_connection(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Event streaming connected",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages (ping/pong for keepalive)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))
                
    except Exception as e:
        logger.warning(f"Event streaming WebSocket error: {e}")
    finally:
        event_streamer.remove_connection(websocket)
