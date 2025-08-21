"""
Video Processing Queue Service - Sequential video processing
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import uuid

from app.models.video import Video, VideoStatus
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class QueueStatus(Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class QueueItem:
    """Represents a video in the processing queue"""
    id: str
    video: Video
    status: QueueStatus = QueueStatus.WAITING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: Dict[str, int] = field(default_factory=lambda: {
        "downloading": 0,
        "transcribing": 0,
        "analyzing": 0
    })

class VideoProcessingQueue:
    """
    Sequential video processing queue service
    Processes videos one by one to avoid rate limits and resource bottlenecks
    """
    
    def __init__(self, video_status: Dict[str, Dict]):
        self.settings = get_settings()
        self.video_status = video_status
        self.queue: List[QueueItem] = []
        self.current_item: Optional[QueueItem] = None
        self.is_running = False
        self.processing_lock = asyncio.Lock()
        
    async def start(self):
        """Start the video processing queue"""
        logger.info("Starting video processing queue")
        self.is_running = True
        # Start the queue processor in the background
        asyncio.create_task(self._process_queue())
        
    async def stop(self):
        """Stop the video processing queue"""
        logger.info("Stopping video processing queue")
        self.is_running = False
        
    async def add_video(self, video: Video) -> str:
        """
        Add a video to the processing queue
        Returns the queue item ID
        """
        queue_item = QueueItem(
            id=str(uuid.uuid4()),
            video=video
        )
        
        # Initialize status in video_status dict
        self.video_status[video.id] = {
            "status": VideoStatus.PENDING,
            "queue_status": QueueStatus.WAITING.value,
            "queue_position": len(self.queue) + (1 if self.current_item else 0),
            "error": None,
            "progress": queue_item.progress.copy(),
            "created_at": queue_item.created_at.isoformat()
        }
        
        self.queue.append(queue_item)
        logger.info(f"Added video {video.id} to queue (position: {len(self.queue)})")
        
        return queue_item.id
        
    async def add_videos_batch(self, videos: List[Video]) -> List[str]:
        """
        Add multiple videos to the queue for sequential processing
        Returns list of queue item IDs
        """
        queue_item_ids = []
        
        for video in videos:
            queue_item_id = await self.add_video(video)
            queue_item_ids.append(queue_item_id)
            
        logger.info(f"Added {len(videos)} videos to processing queue")
        return queue_item_ids
        
    async def _process_queue(self):
        """Main queue processing loop"""
        while self.is_running:
            try:
                # Check for stuck processing - if we have a current_item for more than 30 minutes, consider it stuck
                if self.current_item and self.current_item.started_at:
                    time_elapsed = (datetime.now() - self.current_item.started_at).total_seconds()
                    # If processing has been ongoing for more than 30 minutes, log a warning and reset
                    if time_elapsed > 1800:  # 30 minutes
                        logger.warning(f"Video {self.current_item.video.id} appears stuck in processing for {time_elapsed/60:.1f} minutes. Resetting queue processor.")
                        # Mark as failed and reset
                        self.current_item.status = QueueStatus.FAILED
                        self.current_item.error = "Processing timeout exceeded"
                        self.current_item.completed_at = datetime.now()
                        
                        # Update video status
                        if self.current_item.video.id in self.video_status:
                            self.video_status[self.current_item.video.id].update({
                                "status": VideoStatus.ERROR,
                                "queue_status": QueueStatus.FAILED.value,
                                "error": "Processing timeout exceeded",
                                "completed_at": self.current_item.completed_at.isoformat()
                            })
                        
                        # Reset current_item to allow next video to be processed
                        self.current_item = None
                
                # Process next video if queue has items and no current processing
                if self.queue and not self.current_item:
                    async with self.processing_lock:
                        if self.queue:  # Double-check after acquiring lock
                            logger.info(f"Taking next video from queue. Queue size: {len(self.queue)}")
                            self.current_item = self.queue.pop(0)
                            # Use try/finally to ensure current_item is reset even if processing fails
                            try:
                                await self._process_video(self.current_item)
                            finally:
                                # Always reset current_item after processing completes or fails
                                logger.info(f"Finished processing video {self.current_item.video.id}. Resetting current_item.")
                                self.current_item = None
                
                # Update queue positions for waiting videos
                await self._update_queue_positions()
                
                # Wait a bit before checking queue again - shorter interval for responsiveness
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in queue processor: {str(e)}", exc_info=True)
                if self.current_item:
                    self.current_item.status = QueueStatus.FAILED
                    self.current_item.error = str(e)
                    self.current_item.completed_at = datetime.now()
                    
                    # Update video status
                    if self.current_item.video.id in self.video_status:
                        self.video_status[self.current_item.video.id].update({
                            "status": VideoStatus.ERROR,
                            "queue_status": QueueStatus.FAILED.value,
                            "error": str(e),
                            "completed_at": self.current_item.completed_at.isoformat()
                        })
                    
                    self.current_item = None
                
                # Wait before retrying
                await asyncio.sleep(5)
                
    async def _process_video(self, queue_item: QueueItem):
        """Process a single video from the queue"""
        try:
            logger.info(f"Starting processing of video {queue_item.video.id}")
            
            # Emit video processing start event
            from app.services.event_streamer import emit_video_event, EventType
            emit_video_event(
                EventType.VIDEO_QUEUED, 
                queue_item.video.id, 
                f"Started processing video: {queue_item.video.url}"
            )
            
            # Update status to processing
            queue_item.status = QueueStatus.PROCESSING
            queue_item.started_at = datetime.now()
            
            # Update video status
            self.video_status[queue_item.video.id].update({
                "status": VideoStatus.PROCESSING,
                "queue_status": QueueStatus.PROCESSING.value,
                "queue_position": 0,
                "started_at": queue_item.started_at.isoformat()
            })
            
            # Import here to avoid circular imports
            from app.services.video_processor import VideoProcessor
            
            # Get video processor instance
            video_processor = VideoProcessor(video_status=self.video_status)
            
            # Process the video
            await video_processor.process_video(queue_item.video)
            
            # Mark as completed
            queue_item.status = QueueStatus.COMPLETED
            queue_item.completed_at = datetime.now()
            
            # Update final status
            self.video_status[queue_item.video.id].update({
                "status": VideoStatus.COMPLETED,
                "queue_status": QueueStatus.COMPLETED.value,
                "completed_at": queue_item.completed_at.isoformat()
            })
            
            logger.info(f"Successfully processed video {queue_item.video.id}")
            
        except Exception as e:
            logger.error(f"Error processing video {queue_item.video.id}: {str(e)}", exc_info=True)
            
            # Mark as failed
            queue_item.status = QueueStatus.FAILED
            queue_item.error = str(e)
            queue_item.completed_at = datetime.now()
            
            # Update video status
            self.video_status[queue_item.video.id].update({
                "status": VideoStatus.ERROR,
                "queue_status": QueueStatus.FAILED.value,
                "error": str(e),
                "completed_at": queue_item.completed_at.isoformat()
            })
            
    async def _update_queue_positions(self):
        """Update queue positions for all waiting videos"""
        for i, queue_item in enumerate(self.queue):
            position = i + (1 if self.current_item else 0)
            if queue_item.video.id in self.video_status:
                self.video_status[queue_item.video.id]["queue_position"] = position
                
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        # Count total processed videos (videos that have completed)
        total_processed = sum(1 for video_id, status_info in self.video_status.items() 
                             if status_info.get("status") == VideoStatus.COMPLETED)
        
        return {
            # Frontend expects these field names
            "queue_size": len(self.queue),
            "is_processing": self.is_running and self.current_item is not None,
            "status": "Processing" if (self.is_running and self.current_item) else "Idle",
            "total_processed": total_processed,
            "current_video": self.current_item.video.id if self.current_item else None,
            
            # Additional detailed info
            "is_running": self.is_running,
            "current_processing": {
                "video_id": self.current_item.video.id if self.current_item else None,
                "status": self.current_item.status.value if self.current_item else None,
                "started_at": self.current_item.started_at.isoformat() if self.current_item and self.current_item.started_at else None
            },
            "waiting_videos": [
                {
                    "video_id": item.video.id,
                    "url": str(item.video.url),
                    "position": i + 1,
                    "created_at": item.created_at.isoformat()
                }
                for i, item in enumerate(self.queue)
            ]
        }
        
    def get_video_queue_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get queue information for a specific video"""
        if video_id in self.video_status:
            return {
                "video_id": video_id,
                "queue_status": self.video_status[video_id].get("queue_status"),
                "queue_position": self.video_status[video_id].get("queue_position", 0),
                "created_at": self.video_status[video_id].get("created_at"),
                "started_at": self.video_status[video_id].get("started_at"),
                "completed_at": self.video_status[video_id].get("completed_at")
            }
        return None

# Global queue instance - will be initialized in main.py
video_queue: Optional[VideoProcessingQueue] = None

def get_video_queue() -> VideoProcessingQueue:
    """Get the global video queue instance"""
    global video_queue
    if video_queue is None:
        raise RuntimeError("Video queue not initialized. Call initialize_video_queue() first.")
    return video_queue

def initialize_video_queue(video_status: Dict[str, Dict]) -> VideoProcessingQueue:
    """Initialize the global video queue instance"""
    global video_queue
    video_queue = VideoProcessingQueue(video_status)
    return video_queue