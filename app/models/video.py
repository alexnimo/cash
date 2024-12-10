from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Tuple
from datetime import datetime
from enum import Enum

class VideoSource(str, Enum):
    YOUTUBE = "youtube"

class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoMetadata(BaseModel):
    title: str
    duration: float
    resolution: Tuple[int, int]
    format: str
    size_bytes: int

class TranscriptSegment(BaseModel):
    start_time: float
    end_time: float
    text: str
    frame_idx: Optional[int] = None

class VideoAnalysis(BaseModel):
    transcript: str
    summary: str
    key_frames: List[str]  # Paths to extracted key frames
    embedding_id: str
    
class Video(BaseModel):
    id: str
    source: VideoSource
    url: Optional[HttpUrl] = None
    file_path: Optional[str] = None
    audio_path: Optional[str] = None
    status: VideoStatus = VideoStatus.PENDING
    created_at: datetime = datetime.now()
    metadata: Optional[VideoMetadata] = None
    transcript: Optional[List[TranscriptSegment]] = None
    analysis: Optional[VideoAnalysis] = None
    error: Optional[str] = None
