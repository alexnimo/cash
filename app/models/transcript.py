from typing import List, Optional
from pydantic import BaseModel

class TranscriptSegment(BaseModel):
    """A segment of transcript with timing information and associated frames"""
    start_time: float
    end_time: float
    text: str
    frames: Optional[List[str]] = []  # List of frame filenames associated with this segment
