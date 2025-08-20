import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
import os
from pathlib import Path
import logging
import xml.etree.ElementTree
import xml.parsers.expat
import yt_dlp
from datetime import datetime
import tempfile
import shutil
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
from pytubefix import YouTube
from pytubefix.exceptions import PytubeFixError
# Removed langtrace import
from app.utils.path_utils import get_storage_subdir, get_storage_path
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from app.models.video import Video, VideoSource, VideoStatus, VideoMetadata, VideoAnalysis
from app.core.unified_config import get_config
from app.services.model_manager import ModelManager
from app.services.model_config import configure_models
from app.services.content_analyzer import ContentAnalyzer
from app.services.errors import VideoProcessingError
import math
import uuid
import json
from moviepy import VideoFileClip, AudioFileClip
import yt_dlp
import json
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
# Removed langtrace import
from app.utils.path_utils import get_storage_subdir, get_storage_path
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from app.models.video import Video, VideoSource, VideoStatus, VideoMetadata, VideoAnalysis
from app.core.unified_config import get_config
from app.services.model_manager import ModelManager
from app.services.model_config import configure_models
from app.services.content_analyzer import ContentAnalyzer
from app.services.errors import VideoProcessingError
import re

# Use MoviePy for audio processing instead of pydub for Python 3.13 compatibility
try:
    from moviepy import AudioFileClip
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    AudioFileClip = None

# Keep pydub import for fallback (if available)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

class TranscriptSegment:
    """Represents a segment of a transcript with timestamp information."""
    
    def __init__(self, start_time: float, end_time: float, text: str, language: str = None):
        self.start = start_time
        self.end = end_time
        self.text = text
        self.language = language
    
    def to_dict(self):
        """Convert segment to dictionary representation."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "duration": self.end - self.start,
            "language": self.language
        }

logger = logging.getLogger(__name__)
config = get_config()

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# LangTrace initialization removed
logger.info("LangTrace functionality has been removed from this module")

# Configure models
configure_models()

def is_response_truncated(response: GenerateContentResponse) -> bool:
    """
    Detects if a response from Gemini API was truncated due to token limits.
    
    Args:
        response: The raw response object from Gemini API
        
    Returns:
        bool: True if the response was truncated, False otherwise
    """
    try:
        # Check if response was truncated due to MAX_TOKENS
        finish_reason = response.candidates[0].finish_reason
        
        # Check if the finish_reason.name is "MAX_TOKENS"
        is_truncated = hasattr(finish_reason, 'name') and finish_reason.name == "MAX_TOKENS"
        
        if is_truncated:
            logger.info("Response was truncated due to token limit (FINISH_REASON_MAX_TOKENS)")
        else:
            logger.info(f"Response finish reason: {finish_reason} (not truncated)")
        
        return is_truncated
    except (AttributeError, IndexError) as e:
        logger.warning(f"Could not determine if response was truncated: {str(e)}")
        return False

async def handle_chunked_response(
    model: Any, 
    original_prompt: str, 
    first_response: GenerateContentResponse, 
    content_generator_kwargs: Dict[str, Any] = None,
    max_continuation_attempts: int = 5,
    additional_contents: List[Dict[str, Any]] = None,
    is_json: bool = False
) -> str:
    """
    Handles chunked responses by continuing the generation until complete.
    
    Args:
        model: The Gemini model instance
        original_prompt: The original prompt sent to the model
        first_response: The first (potentially truncated) response from the model
        content_generator_kwargs: Additional kwargs for model.generate_content
        max_continuation_attempts: Maximum number of continuation attempts
        additional_contents: Additional content items (like file_data) to include in continuation requests
        is_json: Whether the response is expected to be JSON format
        
    Returns:
        str: The complete response text
    """
    if not content_generator_kwargs:
        content_generator_kwargs = {}
        
    # Get the text from the first response
    combined_text = first_response.text.strip()
    
    # Check if response is truncated
    if not is_response_truncated(first_response):
        print(f"{GREEN}✓ Response is complete (not truncated){RESET}")
        return combined_text
    
    print(f"{YELLOW}! Initial response is truncated. Starting continuation process...{RESET}")
    
    # Initialize for continuation
    current_response = first_response
    continuation_attempts = 0
    
    # Continue generating content until the response is complete or max attempts reached
    while is_response_truncated(current_response) and continuation_attempts < max_continuation_attempts:
        continuation_attempts += 1
        print(f"{YELLOW}! Making continuation request (attempt {continuation_attempts}/{max_continuation_attempts}){RESET}")
        logger.info(f"Making continuation request (attempt {continuation_attempts}/{max_continuation_attempts})")
        
        # Build continuation prompt
        continuation_prompt = (
            f"""
            Continue exactly from where you have stopped in the following provided response
            add only the missing data, exactly from the point where you have stopped,
            pay attention if there is a missing charecters in the provided response that must be added such
            a missing quate " or colon - this is crucial to complete the previous chunked response in case it's missing
            don't add any greetings or extra messages, follow the original prompt\n\n
            Original prompt:\n{original_prompt}\n\n
            Previous truncated response:\n{combined_text}   
            """
        )
        
        try:
            # Prepare contents for continuation
            contents = [{"text": continuation_prompt}]
            
            # Add any additional file data or other contents
            if additional_contents:
                contents.extend(additional_contents)
                logger.info(f"Including {len(additional_contents)} additional content items in continuation request")
            
            # Generate continuation
            current_response = await asyncio.to_thread(
                model.generate_content,
                contents=contents,
                **content_generator_kwargs
            )
            
            # Check for finish reason
            finish_reason = None
            if hasattr(current_response, "candidates") and current_response.candidates:
                finish_reason = current_response.candidates[0].finish_reason
                print(f"{BLUE}ℹ Continuation {continuation_attempts} finish reason: {finish_reason}{RESET}")
            
            # Append to combined text
            continuation_text = current_response.text.strip()
            
            # Check if we're dealing with JSON content structure
            if is_json:
                logger.info("Detected JSON-like content, using smart continuations")
                
                # 1. Clean up continuation text (remove leading markers)
                if continuation_text.startswith('['):
                    logger.info("Removing leading '[' from continuation response")
                    continuation_text = continuation_text[1:]
                
                # 2. Analyze combined structure for proper joining
                try:
                    # Identify the last complete valid JSON element
                    last_obj_or_array_end = -1
                    for i in range(len(combined_text) - 1, -1, -1):
                        if combined_text[i] in ']}' and combined_text[i-1] not in '\\"':
                            last_obj_or_array_end = i
                            break
                    
                    # Find whether there's a pending comma or if we need to add one
                    # First check if we're inside an array or object
                    brackets_stack = []
                    for char in combined_text:
                        if char in '[{':
                            brackets_stack.append(char)
                        elif char in ']}' and brackets_stack:
                            brackets_stack.pop()
                    
                    outermost_open = brackets_stack[0] if brackets_stack else None
                    inside_collection = bool(brackets_stack)
                    
                    # Trim the combined text if needed to ensure proper continuation
                    if last_obj_or_array_end > 0:
                        # Find the next non-whitespace character after the end
                        next_char_pos = -1
                        for i in range(last_obj_or_array_end + 1, len(combined_text)):
                            if not combined_text[i].isspace():
                                next_char_pos = i
                                break
                        
                        # If the next char is a comma, we need to keep it
                        if next_char_pos != -1 and combined_text[next_char_pos] == ',':
                            combined_text = combined_text[:next_char_pos + 1].rstrip()
                            logger.info("Trimmed to last complete JSON element with trailing comma")
                        # If the next char is closing bracket/brace, we're done with the JSON
                        elif next_char_pos != -1 and combined_text[next_char_pos] in ']}': 
                            combined_text = combined_text[:next_char_pos + 1].rstrip()
                            logger.info("Combined text is already a complete JSON structure")
                            # No need to continue since we have complete JSON
                            break  
                        else:
                            # Otherwise, keep only up to the last complete element
                            combined_text = combined_text[:last_obj_or_array_end + 1].rstrip()
                            logger.info("Trimmed to last complete JSON element")
                            
                            # Add a comma if we're inside a collection and don't already end with one
                            if inside_collection and not combined_text.endswith(',') and not combined_text.endswith(outermost_open):
                                combined_text += ','
                                logger.info("Added comma to prepare for continuation")
                except Exception as e:
                    logger.warning(f"Error during JSON structure analysis: {str(e)}")
                
                # 3. Join the two pieces smartly
                # Check if the continuation starts with a comma or needs one
                continuation_text = continuation_text.lstrip()
                if continuation_text.startswith(','):
                    # Avoid duplicate commas
                    if combined_text.endswith(','):
                        continuation_text = continuation_text[1:].lstrip()
                        logger.info("Removed leading comma from continuation to avoid duplicates")
                # Or maybe we need to add one
                elif not combined_text.endswith(',') and inside_collection and not combined_text.endswith(outermost_open):
                    # Don't add comma if continuation is closing the collection
                    if not continuation_text.strip().startswith(']') and not continuation_text.strip().startswith('}'):
                        continuation_text = ',' + continuation_text
                        logger.info("Added comma between elements for proper JSON continuation")
            
            # Now append in all cases
            combined_text += continuation_text
            
            print(f"{GREEN}✓ Added continuation {continuation_attempts} (length: {len(continuation_text)} chars){RESET}")
            logger.info(f"Added continuation {continuation_attempts} (length: {len(continuation_text)})")
            
            # If not truncated, we're done
            if not is_response_truncated(current_response):
                print(f"{GREEN}✓ Response continuation complete (not truncated){RESET}")
                logger.info("Response continuation complete (not truncated)")
                break
            else:
                print(f"{YELLOW}! Continuation {continuation_attempts} was still truncated, continuing...{RESET}")
                
        except Exception as e:
            print(f"{YELLOW}✗ Error during continuation attempt {continuation_attempts}: {str(e)}{RESET}")
            logger.error(f"Error during continuation attempt {continuation_attempts}: {str(e)}")
            break
    
    return combined_text

class VideoProcessor:
    def __init__(self, video_status: Dict[str, Dict] = None):
        try:
            # Use the unified storage path utilities
            self.video_dir = get_storage_subdir("videos")
            self.audio_dir = get_storage_subdir("videos/audio")
            self.transcript_dir = get_storage_subdir("videos/transcripts")
            self.raw_transcript_dir = get_storage_subdir("videos/raw_transcripts")
            self.frames_dir = get_storage_subdir("videos/frames")
            
            # Create trace directory if langtrace is enabled
            if config.langtrace and getattr(config.langtrace, 'enabled', False):
                trace_dir = Path(config.langtrace.trace_dir)
                if not trace_dir.is_absolute():
                    trace_dir = get_storage_path(config.langtrace.trace_dir)
                    trace_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize services
            self.model_manager = ModelManager(config)
            self.content_analyzer = ContentAnalyzer()
            
            # Initialize semaphore for parallel processing
            max_chunks = config.processing.max_parallel_chunks if hasattr(config, 'processing') and hasattr(config.processing, 'max_parallel_chunks') else 3
            self.semaphore = asyncio.Semaphore(max_chunks)
            logger.info(f"Initialized semaphore with limit of {max_chunks} concurrent tasks")
            
            # Store reference to global video status
            self.video_status = video_status if video_status is not None else {}
            
            logger.info(f"Initialized VideoProcessor with video directory: {self.video_dir}")
            
        except Exception as e:
            raise VideoProcessingError(f"Failed to initialize VideoProcessor: {str(e)}")

    async def _get_youtube_transcript(self, video: Video) -> Optional[str]:
        """
        Attempt to get the transcript directly from YouTube.
        First tries to get English transcript, if not available gets any available transcript.
        Returns the transcript text if available, None otherwise.
        """
        try:
            url = str(video.url)
            logger.info(f"Attempting to get transcript for video URL: {url}")
            
            # Extract video ID from URL
            video_id = None
            if 'youtube.com' in url or 'youtu.be' in url:
                if 'youtube.com/watch?v=' in url:
                    video_id = url.split('watch?v=')[1].split('&')[0]
                elif 'youtu.be/' in url:
                    video_id = url.split('youtu.be/')[1].split('?')[0]
            
            if not video_id:
                logger.warning(f"Could not extract YouTube video ID from URL: {url}")
                return None

            logger.info(f"Extracted YouTube video ID: {video_id}")

            # Create API instance
            api = YouTubeTranscriptApi()
            
            # Try to get English transcript first
            logger.info("Trying to get English transcript...")
            try:
                fetched_transcript = api.fetch(video_id, languages=['en'])
                # The fetch method already returns a FetchedTranscript with transcript data
                transcript_data = fetched_transcript
                transcript_language = 'en'
                logger.info("Found English transcript")
            except NoTranscriptFound:
                # If English not available, try any language
                logger.info("No English transcript found, trying any available transcript...")
                try:
                    # List all available transcripts first
                    transcript_list = api.list(video_id)
                    if transcript_list:
                        # Get the first available transcript
                        available = list(transcript_list)
                        if available:
                            first_transcript = available[0]
                            transcript_data = first_transcript.fetch()
                            transcript_language = first_transcript.language_code
                            logger.info(f"Using available transcript in language: {transcript_language}")
                        else:
                            logger.warning("No transcripts available")
                            return None
                    else:
                        logger.warning("No transcripts available")
                        return None
                except NoTranscriptFound:
                    logger.warning("No transcripts available")
                    return None
                except Exception as e:
                    logger.error(f"Error getting any transcript: {str(e)}")
                    return None

            if transcript_data:
                # Process the transcript data
                logger.info(f"Processing transcript data in {transcript_language}...")
                logger.info(f"Got transcript with {len(transcript_data)} segments")
                
                # Handle FetchedTranscriptSnippet objects
                # Extract data in a way that works with both dictionary-like objects and FetchedTranscriptSnippet objects
                processed_segments = []
                for segment in transcript_data:
                    # Try to access attributes as properties first, then fall back to dictionary access
                    try:
                        # If it's a FetchedTranscriptSnippet object with properties
                        segment_dict = {
                            "text": getattr(segment, "text", ""),
                            "start": getattr(segment, "start", 0.0),
                            "duration": getattr(segment, "duration", 0.0)
                        }
                    except Exception:
                        # If it's a dictionary-like object
                        try:
                            segment_dict = {
                                "text": segment.get("text", ""),
                                "start": segment.get("start", 0.0),
                                "duration": segment.get("duration", 0.0)
                            }
                        except Exception:
                            # Last resort, try direct dictionary access
                            try:
                                segment_dict = {
                                    "text": segment["text"] if "text" in segment else "",
                                    "start": segment["start"] if "start" in segment else 0.0,
                                    "duration": segment["duration"] if "duration" in segment else 0.0
                                }
                            except Exception as e:
                                logger.warning(f"Could not extract data from transcript segment: {e}")
                                segment_dict = {"text": "", "start": 0.0, "duration": 0.0}
                    
                    processed_segments.append(segment_dict)
                
                # Save raw transcript text
                raw_text = ' '.join(segment["text"] for segment in processed_segments)
                raw_transcript_path = get_storage_path(f"videos/raw_transcripts/{video.id}.txt")
                with open(raw_transcript_path, 'w', encoding='utf-8') as f:
                    f.write(raw_text)
                logger.info(f"Saved raw transcript to {raw_transcript_path}")
                
                # Save structured transcript to file
                transcript_path = get_storage_path(f"videos/transcripts/{video.id}.json")
                logger.info(f"Saving transcript to {transcript_path}")
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "language": transcript_language,
                        "channel_name": video.metadata.channel_name if video.metadata else "",
                        "video_name": video.metadata.video_title if video.metadata else "",
                        "segments": processed_segments
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Successfully saved transcript to {transcript_path}")
                return str(transcript_path)
                
        except TranscriptsDisabled:
            logger.error("Transcripts are disabled for this video")
            return None
        except NoTranscriptFound:
            logger.error("No transcript found for this video")
            return None
        except Exception as e:
            logger.error(f"Error getting YouTube transcript: {str(e)}", exc_info=True)
            return None
            
    def __del__(self):
        """Synchronous cleanup - async cleanup should be done explicitly via cleanup() method"""
        try:
            # Clean up any semaphores synchronously
            if hasattr(self, 'semaphore'):
                # Release any acquired semaphores
                for _ in range(getattr(self.semaphore, '_value', 0)):
                    try:
                        self.semaphore.release()
                    except (ValueError, AttributeError):
                        # Stop if we've released too many or semaphore is gone
                        break
                logger.debug("Cleaned up semaphores during VideoProcessor deletion")
        except Exception as e:
            logger.error(f"Error in VideoProcessor cleanup: {e}")
    
    async def cleanup(self):
        """Async cleanup method for proper resource management"""
        try:
            # Close any open resources
            if hasattr(self, 'model_manager') and self.model_manager is not None:
                if hasattr(self.model_manager, 'close'):
                    await self.model_manager.close()
            logger.debug("VideoProcessor async cleanup completed")
        except Exception as e:
            logger.error(f"Error in VideoProcessor async cleanup: {e}")

    async def process_video(self, video: Video) -> Video:
        """Process a video by downloading it and generating a transcript."""
        # Track resources that need cleanup
        resources_to_cleanup = []
        
        try:
            logger.info(f"Starting to process video {video.id} from {video.url}")
            
            # Emit event for processing start
            from app.services.event_streamer import emit_video_event, EventType
            emit_video_event(
                EventType.VIDEO_QUEUED, 
                video.id, 
                f"Starting to process video: {video.url}"
            )
            
            # Update status
            self.video_status[video.id] = {
                "status": "processing",
                "progress": {
                    "downloading": 0,
                    "transcribing": 0,
                    "analyzing": 0,
                    "extracting_frames": 0
                },
                "url": str(video.url)
            }
            
            # Reset and recreate semaphore for each video to prevent leaks
            max_chunks = config.processing.max_parallel_chunks if hasattr(config, 'processing') and hasattr(config.processing, 'max_parallel_chunks') else 3
            self.semaphore = asyncio.Semaphore(max_chunks)
            resources_to_cleanup.append('semaphore')
            logger.info(f"Created new semaphore with value {max_chunks}")
            
            # Create a new model manager instance for this video
            if hasattr(self, 'model_manager'):
                # Close existing model manager if there is one
                try:
                    await self.model_manager.close()
                except Exception as e:
                    logger.warning(f"Error closing previous model manager: {e}")
            
            # Create fresh model manager
            self.model_manager = ModelManager(config)
            resources_to_cleanup.append('model_manager')
            logger.info("Created new model manager instance")
            
            # Validate transcription configuration before downloading anything
            try:
                logger.info("Validating transcription configuration before proceeding")
                if not hasattr(self, 'model_manager'):
                    self.model_manager = ModelManager(config)
                
                # Test if the transcription model is properly configured by attempting to access it
                # This will raise an error if configuration is invalid
                await self.model_manager.get_transcription_model()
                logger.info("Transcription configuration validated successfully")
            except Exception as config_error:
                logger.error(f"Transcription configuration validation failed: {str(config_error)}")
                self.video_status[video.id]["status"] = "failed"
                self.video_status[video.id]["error"] = f"Transcription configuration error: {str(config_error)}"
                raise VideoProcessingError(f"Transcription configuration is invalid: {str(config_error)}")

            # Download video
            logger.info(f"Processing video {video.id}")
            video_path = await self._download_video(video)
            video.file_path = video_path
            logger.info(f"Video downloaded to {video_path}")

            # Extract additional metadata if needed
            if not video.metadata:
                logger.info(f"Extracting metadata for video {video.id}")
                video.metadata = await self._extract_metadata(video_path)
                logger.info(f"Metadata extracted: {video.metadata}")

            # Try to get transcript from YouTube if it's a YouTube video
            transcript_obtained = False
            if video.source == VideoSource.YOUTUBE:
                try:
                    logger.info(f"Attempting to get YouTube transcript for video {video.id}")
                    
                    # Emit transcription start event
                    from app.services.event_streamer import emit_video_event, EventType
                    emit_video_event(
                        EventType.TRANSCRIPTION_START,
                        video.id,
                        "Getting transcript from YouTube..."
                    )
                    transcript_path = await self._get_youtube_transcript(video)
                    if transcript_path:
                        logger.info(f"Successfully retrieved YouTube transcript from {transcript_path}")
                        # Parse the saved transcript file
                        with open(transcript_path) as f:
                            transcript_data = json.load(f)
                            logger.info(f"Loaded transcript data with language: {transcript_data.get('language')}")
                            video.transcript = [TranscriptSegment(
                                start_time=entry.get('start', 0.0),
                                end_time=entry.get('start', 0.0) + entry.get('duration', 0.0),
                                text=entry.get('text', ''),
                                language=transcript_data.get("language", 'en')
                            ) for entry in transcript_data.get("segments", [])]
                            logger.info(f"Created {len(video.transcript)} transcript segments")
                        self.video_status[video.id]["progress"]["transcribing"] = 100
                        logger.info("Using YouTube transcript")
                        transcript_obtained = True
                except Exception as e:
                    logger.error(f"Failed to get YouTube transcript: {str(e)}", exc_info=True)
                    transcript_obtained = False

            # If no YouTube transcript available, proceed with audio extraction and transcription
            if not transcript_obtained:
                try:
                    # Extract audio
                    logger.info(f"Extracting audio from video {video.id}")
                    audio_path = await self._extract_audio(video_path)
                    logger.info(f"Audio extracted to {audio_path}")

                    # Generate transcript
                    logger.info(f"Generating transcript for video {video.id}")
                    video.transcript = await self._generate_transcript(audio_path)
                    self.video_status[video.id]["progress"]["transcribing"] = 100
                    logger.info("Transcript generated successfully")
                    
                    # Check if transcript was successfully generated
                    if not video.transcript or len(video.transcript) == 0:
                        logger.error(f"No transcript was generated for video {video.id}")
                        self.video_status[video.id]["status"] = "failed"
                        self.video_status[video.id]["error"] = "Transcript generation failed"
                        raise VideoProcessingError("No transcript was generated - cannot proceed with analysis")
                        
                except Exception as e:
                    logger.error(f"Failed to generate transcript: {str(e)}")
                    self.video_status[video.id]["status"] = "failed"
                    self.video_status[video.id]["error"] = f"Failed to generate transcript: {str(e)}"
                    raise VideoProcessingError(f"Failed to generate transcript: {str(e)}")
                    
            # Verify we have a valid transcript before proceeding
            if not video.transcript or len(video.transcript) == 0:
                logger.error(f"No valid transcript available for video {video.id}")
                self.video_status[video.id]["status"] = "failed"
                self.video_status[video.id]["error"] = "No valid transcript available"
                raise VideoProcessingError("No valid transcript available - cannot proceed with analysis")
                    
            # Analyze content first - we need the transcript and summary before extracting frames
            logger.info(f"Analyzing content for video {video.id}")
            video = await self._analyze_content(video)
            self.video_status[video.id]["progress"]["analyzing"] = 100
            logger.info("Content analysis completed")
            
            # If content analysis failed, don't proceed with frame extraction
            if video.status == VideoStatus.FAILED:
                logger.warning(f"Skipping frame extraction for video {video.id} due to content analysis failure")
                self.video_status[video.id]["progress"]["extracting_frames"] = 0
            else:
                # Extract frames only if transcript processing succeeded
                logger.info(f"Extracting frames from video {video.id}")
                frames = await self._extract_frames(video)
                logger.debug(f"Extracted frames: {frames}")
                
                if frames:
                    # Update analysis with frame paths
                    if not video.analysis:
                        logger.debug("Creating new VideoAnalysis with extracted frames")
                        video.analysis = VideoAnalysis(
                            transcript='',
                            summary='',
                            key_frames=frames,
                            embedding_id=''
                        )
                    else:
                        logger.debug(f"Updating existing VideoAnalysis with {len(frames)} frames")
                        video.analysis.key_frames = frames
                    logger.info(f"Extracted {len(frames)} frames")
                    self.video_status[video.id]["progress"]["extracting_frames"] = 100
                    
                    # IMPORTANT: After frames are extracted, we need to re-run content analysis 
                    # to process the frames and update the analysis with frame information
                    logger.info(f"Re-running content analysis to include frame analysis for video {video.id}")
                    video = await self._analyze_content(video)
                    logger.info("Frame analysis completed and included in content analysis")
                else:
                    logger.warning("No frames were extracted")
                    self.video_status[video.id]["progress"]["extracting_frames"] = 0
                
                # Agent workflow - integrated into standard processing
                logger.info(f"Starting integrated agent workflow for video {video.id}")
                try:
                    # Execute agent workflow as an integral part of the processing pipeline
                    result = await self._trigger_agent_workflow(video)
                    logger.info(f"Agent workflow completed with result: {result}")
                except Exception as agent_error:
                    logger.error(f"Failed to complete agent workflow: {str(agent_error)}", exc_info=True)
                    # Continue with video processing even if agent workflow fails

            # Update status
            self.video_status[video.id]["status"] = "completed"
            logger.info(f"Video {video.id} processed successfully")

            # Emit video processing completion event
            from app.services.event_streamer import emit_video_event, EventType
            emit_video_event(
                EventType.VIDEO_PROCESSING_COMPLETE, 
                video.id, 
                "Video processing completed successfully!"
            )

            return video

        except VideoProcessingError as e:
            # This is an expected error, log it normally
            logger.error(f"Video processing error: {str(e)}")
            self.video_status[video.id]["status"] = "error"
            self.video_status[video.id]["error"] = str(e)
            
            # Emit error event
            from app.services.event_streamer import emit_video_event, EventType
            emit_video_event(
                EventType.ERROR_OCCURRED, 
                video.id, 
                f"Processing error: {str(e)}"
            )
            
            raise
            
        except Exception as e:
            # Unexpected error, log with traceback
            logger.exception(f"Unexpected error processing video {video.id}: {str(e)}")
            self.video_status[video.id]["status"] = "error"
            self.video_status[video.id]["error"] = str(e)
            
            # Emit error event
            from app.services.event_streamer import emit_video_event, EventType
            emit_video_event(
                EventType.ERROR_OCCURRED, 
                video.id, 
                f"Unexpected error: {str(e)}"
            )
            
            raise VideoProcessingError(f"Unexpected error: {str(e)}") from e
            
        finally:
            # Always clean up resources
            logger.info(f"Cleaning up resources for video {video.id}")
            
            # Clean up any open video/audio files
            for temp_path in [v for v in getattr(self, '_temp_files', []) if v]:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        logger.debug(f"Removed temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_path}: {e}")
            
            # Reset temp files list
            self._temp_files = []
            
            # Clean up model resources
            if hasattr(self, 'model_manager'):
                try:
                    await self.model_manager.close()
                    logger.debug("Closed model manager resources")
                except Exception as e:
                    logger.warning(f"Error closing model manager: {e}")
            
            # Reset semaphores
            if hasattr(self, 'semaphore'):
                try:
                    # Release any acquired permits
                    current = self.semaphore._value
                    max_value = getattr(config.processing, 'max_parallel_chunks', 3)
                    for _ in range(max_value - current):
                        try:
                            self.semaphore.release()
                        except ValueError:
                            # Stop if we've released too many
                            break
                    logger.debug("Reset semaphore resources")
                except Exception as e:
                    logger.warning(f"Error resetting semaphore: {e}")

    async def process_playlist(self, url: str) -> List[str]:
        """Process a YouTube playlist and return list of video IDs"""
        try:
            logger.info(f"Processing playlist: {url}")
            
            # Configure yt-dlp for playlist
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,  # Only extract video URLs without downloading
                'source_address': '127.0.0.1',  # Force ipv4 to avoid getting 403 errors
                'cookiefile' : 'cookies.txt',
            }
            
            # Extract video URLs from playlist
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' not in info:
                    raise VideoProcessingError("No videos found in playlist")
                    
                video_urls = []
                for entry in info['entries']:
                    if entry and 'url' in entry:
                        video_urls.append(entry['url'])
            
            logger.info(f"Found {len(video_urls)} videos in playlist")
            return video_urls
            
        except Exception as e:
            logger.error(f"Failed to process playlist: {str(e)}")
            raise VideoProcessingError(f"Failed to process playlist: {str(e)}")

    async def _download_video(self, video: Video) -> str:
        """Download video from YouTube"""
        try:
            video_url = str(video.url)
            logger.info(f"Downloading video from URL: {video_url}")
            
            # Enhanced configuration for metadata extraction to avoid bot detection
            metadata_opts = {
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': False,
                'nocheckcertificate': True,
                # Add headers to avoid bot detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Connection': 'keep-alive',
                },
                # Additional options to avoid restrictions
                'extractor_retries': 3,
                'fragment_retries': 3,
                'retry_sleep_functions': {'http': lambda n: min(4 ** n, 60)},
            }
            
            with yt_dlp.YoutubeDL(metadata_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if info is None:
                    raise VideoProcessingError("Failed to get video metadata")
                
                # Log all potential metadata fields for debugging
                metadata_fields = {
                    'uploader': info.get('uploader'),
                    'channel': info.get('channel'),
                    'channel_id': info.get('channel_id'),
                    'uploader_id': info.get('uploader_id'),
                    'uploader_url': info.get('uploader_url'),
                    'channel_url': info.get('channel_url'),
                    'title': info.get('title'),
                    'description': info.get('description')
                }
                logger.info(f"Available metadata fields: {metadata_fields}")
                
                # Try different fields for channel name
                channel_name = (
                    info.get('uploader') or 
                    info.get('channel') or 
                    info.get('channel_id') or 
                    info.get('uploader_id') or
                    info.get('uploader_url', '').split('/')[-1] or
                    info.get('channel_url', '').split('/')[-1] or
                    ''  # Empty string if no channel info available
                )
                
                # Extract metadata before download
                video.metadata = VideoMetadata(
                    title=info.get('title', ''),
                    duration=float(info.get('duration', 0)),
                    resolution=(
                        info.get('width', 0),
                        info.get('height', 0)
                    ),
                    format='mp4',
                    size_bytes=info.get('filesize', 0),
                    channel_name=channel_name,
                    video_title=info.get('title', '')
                )
                
                logger.info(f"Processed metadata - Channel: {channel_name}, Title: {video.metadata.video_title}")
            
            # Download video with comprehensive anti-bot measures
            video_opts = {
                # Most flexible format selection - try anything available
                'format': 'worst/best',  # Start with worst quality to avoid premium formats
                'outtmpl': str(self.video_dir / f'{video.id}.%(ext)s'),
                'progress_hooks': [lambda d: self._update_download_progress(video.id, d)],
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': False,
                'nocheckcertificate': True,
                # Enhanced headers to mimic real browser
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0',
                },
                # Aggressive retry and delay settings
                'extractor_retries': 5,
                'fragment_retries': 5,
                'retry_sleep_functions': {'http': lambda n: min(2 ** n, 30)},
                'sleep_interval': 1,
                'max_sleep_interval': 5,
                # Additional YouTube-specific options
                'youtube_include_dash_manifest': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            
            # Download video
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if info is None:
                    raise VideoProcessingError("Failed to download video")
                
                video_path = ydl.prepare_filename(info)
                if not os.path.exists(video_path):
                    potential_files = list(self.video_dir.glob(f'{video.id}.*'))
                    if not potential_files:
                        raise VideoProcessingError(f"Downloaded video file not found for ID: {video.id}")
                    video_path = str(potential_files[0])
                
                logger.info(f"Successfully downloaded video to: {video_path}")
                return video_path
            
        except Exception as e:
            logger.error(f"Failed to download video with yt-dlp: {str(e)}")
            fallback_triggers = [
                "403",
                "Requested format is not available", 
                "Failed to extract any player response",
                "Sign in to confirm your age",
                "Private video",
                "Video unavailable"
            ]
            
            should_fallback = any(trigger in str(e) for trigger in fallback_triggers)
            
            if should_fallback:
                logger.info(f"Attempting fallback to pytubefix due to yt-dlp failure: {str(e)}")
                try:
                    return await self._download_video_pytubefix_fallback(video)
                except Exception as fallback_error:
                    logger.error(f"Pytubefix fallback also failed: {str(fallback_error)}")
                    raise VideoProcessingError(f"Both yt-dlp and pytubefix failed. yt-dlp: {str(e)}, pytubefix: {str(fallback_error)}")
            else:
                raise VideoProcessingError(f"Failed to download video: {str(e)}")

    async def _download_video_pytubefix_fallback(self, video: Video) -> str:
        """Fallback method using pytubefix when yt-dlp fails"""
        try:
            video_url = str(video.url)
            logger.info(f"Using pytubefix fallback for URL: {video_url}")
            
            # Create YouTube object with OAuth for better reliability
            yt = YouTube(
                video_url,
                use_oauth=False,  # Start without OAuth, can be enabled if needed
                allow_oauth_cache=True
            )
            
            # Extract metadata if not already done
            if not video.metadata:
                # Get channel name from various sources
                channel_name = (
                    getattr(yt, 'author', None) or 
                    getattr(yt, 'channel_name', None) or
                    ''
                )
                
                video.metadata = VideoMetadata(
                    title=yt.title or '',
                    duration=float(yt.length or 0),
                    resolution=(
                        getattr(yt.streams.get_highest_resolution(), 'resolution', '720p').replace('p', ''),
                        720  # Default height
                    ),
                    format='mp4',
                    size_bytes=0,  # Will be updated after download
                    channel_name=channel_name,
                    video_title=yt.title or ''
                )
                
                logger.info(f"Extracted metadata with pytubefix - Channel: {channel_name}, Title: {video.metadata.video_title}")
            
            # Get the best available stream (prefer mp4)
            stream = None
            
            # Try different stream selection strategies
            strategies = [
                lambda: yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first(),
                lambda: yt.streams.filter(adaptive=True, file_extension='mp4', only_video=True).order_by('resolution').desc().first(),
                lambda: yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first(),
                lambda: yt.streams.get_highest_resolution(),
                lambda: yt.streams.first()
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    stream = strategy()
                    if stream:
                        logger.info(f"Selected stream using strategy {i+1}: {stream.resolution} {stream.mime_type}")
                        break
                except Exception as strategy_error:
                    logger.debug(f"Strategy {i+1} failed: {strategy_error}")
                    continue
            
            if not stream:
                raise VideoProcessingError("No suitable video stream found with pytubefix")
            
            # Download the video
            output_path = str(self.video_dir)
            filename = f"{video.id}.{stream.subtype}"
            
            logger.info(f"Downloading with pytubefix: {stream.resolution} {stream.mime_type}")
            
            # Download with progress callback
            def progress_callback(stream, chunk, bytes_remaining):
                total_size = stream.filesize
                bytes_downloaded = total_size - bytes_remaining
                percentage = int((bytes_downloaded / total_size) * 100)
                self._update_download_progress(video.id, {
                    'status': 'downloading',
                    '_percent_str': f'{percentage}%',
                    'downloaded_bytes': bytes_downloaded,
                    'total_bytes': total_size
                })
            
            # Set progress callback
            yt.register_on_progress_callback(progress_callback)
            
            # Download the file
            downloaded_file = stream.download(
                output_path=output_path,
                filename=filename
            )
            
            if not os.path.exists(downloaded_file):
                raise VideoProcessingError(f"Downloaded file not found: {downloaded_file}")
            
            # Update metadata with actual file size
            if video.metadata:
                video.metadata.size_bytes = os.path.getsize(downloaded_file)
            
            logger.info(f"Successfully downloaded video with pytubefix to: {downloaded_file}")
            return downloaded_file
            
        except PytubeFixError as e:
            logger.error(f"PytubeFixError: {str(e)}")
            raise VideoProcessingError(f"Pytubefix error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to download video with pytubefix: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Pytubefix fallback failed: {str(e)}")

    async def _validate_video_file(self, file_path: str) -> bool:
        """Validate video file using ffmpeg"""
        try:
            import ffmpeg
            probe = ffmpeg.probe(file_path)
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return video_info is not None
        except Exception as e:
            logger.error(f"Failed to validate video file: {str(e)}")
            return False

    def _update_download_progress(self, video_id: str, d: dict):
        """Update download progress"""
        try:
            if d['status'] == 'downloading':
                # Calculate download percentage
                total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                if total_bytes > 0:
                    downloaded = d.get('downloaded_bytes', 0)
                    progress = (downloaded / total_bytes) * 100
                    self.video_status[video_id]["progress"]["downloading"] = progress
                    logger.debug(f"Download progress for video {video_id}: {progress:.1f}%")
            elif d['status'] == 'finished':
                self.video_status[video_id]["progress"]["downloading"] = 100
                logger.info(f"Download completed for video {video_id}")
        except Exception as e:
            logger.error(f"Error updating download progress: {str(e)}")

    async def _extract_audio(self, video_path: str) -> str:
        """Download audio from video URL"""
        try:
            video_id = Path(video_path).stem
            audio_path = get_storage_path(f"videos/audio/{video_id}.wav")
            
            # Get video URL from the video status
            video_url = None
            for vid_id, status in self.video_status.items():
                if vid_id == video_id:
                    video_url = status.get('url')
                    break
            
            if not video_url:
                # If URL not found in status, try to use the original video path
                video_url = str(video_path)
            
            logger.info(f"Downloading audio from URL: {video_url}")
            
            # Download audio separately
            audio_opts = {
                'format': 'bestaudio/wav',  # Get best audio
                'outtmpl': str(self.audio_dir / f'{video_id}.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': False,
                'nocheckcertificate': True,
                'geo_bypass': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }]
            }
            
            # Download audio
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if info is None:
                    raise VideoProcessingError("Failed to download audio")
                
                # Check for audio file
                if not os.path.exists(audio_path):
                    potential_audio_files = list(self.audio_dir.glob(f'{video_id}.*'))
                    if not potential_audio_files:
                        raise VideoProcessingError(f"Downloaded audio file not found for ID: {video_id}")
                    audio_path = str(potential_audio_files[0])
                
                logger.info(f"Successfully downloaded audio to: {audio_path}")
                return audio_path
            
        except Exception as e:
            logger.error(f"Failed to download audio: {str(e)}")
            raise VideoProcessingError(f"Failed to extract audio: {str(e)}")

    async def _process_audio_chunk(self, chunk_path: Path) -> Optional[str]:
        """Process a single audio chunk for transcription.
        
        Args:
            chunk_path: Path to the audio chunk file
            
        Returns:
            Optional[str]: Transcribed text or None if failed
        """
        logger.info(f"Processing audio chunk: {chunk_path}")
        try:
            # Acquire semaphore to control parallel processing
            async with self.semaphore:
                logger.info(f"Acquired semaphore for processing chunk: {chunk_path}")
                
                # Initialize model manager if needed
                if not hasattr(self, 'model_manager'):
                    self.model_manager = ModelManager(config)
                    
                # Upload the audio file
                uploaded_file = await asyncio.to_thread(genai.upload_file, chunk_path, mime_type="audio/wav")
                logger.info(f"Uploaded audio chunk for processing: {uploaded_file.name}")
                
                # Create the transcription prompt
                prompt = "Please provide a complete and accurate transcription of this audio. Maintain the original meaning and include all spoken content."
                
                # Get the model - use the synchronous version to avoid coroutine issues
                model = self.model_manager.get_transcription_model_instance()
                
                # Generate content with audio input
                response = await asyncio.to_thread(
                    model.generate_content, 
                    [
                        prompt,
                        uploaded_file
                    ],
                    generation_config={
                        "temperature": 0.2,
                    }
                )
                
                # Handle potentially chunked responses (if the transcription is long)
                result = await handle_chunked_response(
                    model=model,
                    original_prompt=prompt,
                    first_response=response,
                    additional_contents=[uploaded_file],
                    max_continuation_attempts=5
                )
                
                # Delete the uploaded file
                await asyncio.to_thread(genai.delete_file, uploaded_file.name)
                logger.info(f"Deleted uploaded audio chunk: {uploaded_file.name}")
                
                return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk {chunk_path}: {e}")
            return None

    async def _extract_metadata(self, video_path: str) -> VideoMetadata:
        """Extract metadata from video file"""
        try:
            logger.info(f"Extracting metadata from video: {video_path}")
            video = VideoFileClip(video_path)
            
            # If metadata already exists, preserve channel_name and video_title
            existing_metadata = getattr(self, 'metadata', None)
            channel_name = existing_metadata.channel_name if existing_metadata else ""
            video_title = existing_metadata.video_title if existing_metadata else ""
            
            metadata = VideoMetadata(
                title=Path(video_path).stem,
                duration=video.duration,
                resolution=(video.size[0], video.size[1]),
                format=Path(video_path).suffix[1:],
                size_bytes=os.path.getsize(video_path),
                channel_name=channel_name,
                video_title=video_title
            )
            
            video.close()
            logger.info(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to extract metadata: {str(e)}")

   
    async def _generate_transcript(self, audio_path: str) -> List[TranscriptSegment]:
        logger.info(f"DEBUG: Starting transcript generation with audio_path: {audio_path}")
        """Generate transcript from audio file using Gemini with chunking support for long responses"""
        try:
            logger.info(f"Generating transcript for audio {audio_path}")
            
            # Check if transcript exists first
            video_id = Path(audio_path).stem
            transcript_path = get_storage_path(f"videos/transcripts/{video_id}.json")
            if transcript_path.exists():
                logger.info(f"Found existing transcript at {transcript_path}")
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    return [TranscriptSegment(
                        start_time=entry['start'],
                        end_time=entry['start'] + entry['duration'],
                        text=entry['text'],
                        language=transcript_data["language"]
                    ) for entry in transcript_data["segments"]]
            
            # If no transcript exists, generate one
            raw_text = None
            try:
                # Add exponential backoff retry logic
                base_delay = 1  # Start with 1 second delay
                max_retries = 3
                
                # Check if audio processing is available
                if not AUDIO_PROCESSING_AVAILABLE:
                    logger.error("Audio processing library is not available. MoviePy AudioFileClip is required for audio processing")
                    raise ImportError("Audio processing library is required for audio processing")

                if not Path(audio_path).exists():
                    logger.error(f"Audio file not found at path: {audio_path}")
                    return None

                MAX_CHUNK_DURATION_MS = config.get('transcript_generation', 'max_chunk_duration_minutes', default=30) * 60 * 1000
                logger.info(f"Maximum audio chunk duration set to {config.get('transcript_generation', 'max_chunk_duration_minutes', default=30)} minutes ({MAX_CHUNK_DURATION_MS} ms).")

                try:
                    # Use MoviePy to get audio duration
                    audio_clip = await asyncio.to_thread(AudioFileClip, audio_path)
                    duration_ms = int(audio_clip.duration * 1000)  # Convert seconds to milliseconds
                    logger.info(f"Audio duration: {duration_ms / 1000:.2f} seconds")
                    audio_clip.close()  # Important to close the clip
                except Exception as e:
                    logger.error(f"Failed to load audio file with MoviePy: {e}")
                    return None

                # --- Audio Processing Logic --- 
                if duration_ms <= MAX_CHUNK_DURATION_MS:
                    # --- Process shorter audio files (existing logic) --- 
                    logger.info("Audio duration is within limit, processing as a single file.")
                    audio_file = None
                    try:
                        audio_file = await asyncio.to_thread(genai.upload_file, audio_path, mime_type="audio/wav")
                        logger.info(f"Successfully uploaded audio file: {audio_path}")
                        
                        # Get the Gemini model
                        model = self.model_manager.get_transcription_model_instance()
                        
                        # Create transcription prompt
                        transcription_prompt = """
                        Your task is to provide a complete and accurate transcription of the entire audio file.
                        Requirements:
                        1. Transcribe ALL spoken content from start to finish
                        2. Do not skip or summarize any parts
                        3. Format as plain text without timestamps or speaker labels
                        4. Maintain word-for-word accuracy
                        5. Include every single word that is spoken
                        
                        Important: The transcription must be complete and cover the entire duration of the audio.
                        """
                        
                        initial_response = await asyncio.to_thread(
                            model.generate_content, 
                            contents=[transcription_prompt, audio_file],
                            request_options={"timeout": 1000} # Increased timeout for potentially long audio
                        )

                        if is_response_truncated(initial_response):
                            logger.info("Initial transcription response was truncated, using chunking mechanism for text response.")
                            raw_text = await handle_chunked_response(
                                model=model,
                                original_prompt=transcription_prompt,
                                first_response=initial_response,
                                additional_contents=[audio_file],
                                max_continuation_attempts=5
                            )
                        else:
                            raw_text = initial_response.text.strip()
                            logger.info("Transcription complete (single API call, no text truncation).")
                    except Exception as e:
                        logger.error(f"Error during single-file audio transcription: {e}")
                        raw_text = None
                    finally:
                        # Cleanup chunk resources
                        if audio_file:
                            try:
                                await asyncio.to_thread(genai.delete_file, audio_file.name)
                                logger.info(f"Deleted uploaded single audio file: {audio_file.name}")
                            except Exception as delete_err:
                                logger.warning(f"Failed to delete uploaded single audio file {audio_file.name}: {delete_err}")
                else:
                    # --- Process long audio files via segmentation using MoviePy --- 
                    logger.info(f"Audio duration exceeds limit ({duration_ms} > {MAX_CHUNK_DURATION_MS} ms). Splitting into chunks using MoviePy.")
                    num_chunks = math.ceil(duration_ms / MAX_CHUNK_DURATION_MS)
                    logger.info(f"Splitting audio into {num_chunks} chunks.")
                    transcript_parts = []
                    base_filename = Path(audio_path).stem
                    chunk_temp_dir = self.audio_dir / f"chunks_{base_filename}_{uuid.uuid4()}"
                    
                    try:
                        chunk_temp_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created temporary chunk directory: {chunk_temp_dir}")
                        
                        # Load the full audio clip once
                        full_audio_clip = await asyncio.to_thread(AudioFileClip, audio_path)

                        for i in range(num_chunks):
                            start_seconds = (i * MAX_CHUNK_DURATION_MS) / 1000
                            end_seconds = min(((i + 1) * MAX_CHUNK_DURATION_MS) / 1000, duration_ms / 1000)
                            logger.info(f"Processing chunk {i+1}/{num_chunks} ({start_seconds:.2f}s to {end_seconds:.2f}s)")
                            
                            chunk_filename = f"{base_filename}_chunk_{i+1}.wav"
                            chunk_path = chunk_temp_dir / chunk_filename
                            uploaded_chunk_file = None

                            try:
                                # Create audio subclip using MoviePy
                                chunk_clip = full_audio_clip.subclipped(start_seconds, end_seconds)
                                
                                # Export chunk using MoviePy
                                await asyncio.to_thread(chunk_clip.write_audiofile, str(chunk_path), 
                                                       verbose=False, logger=None)
                                logger.info(f"Exported chunk {i+1} to {chunk_path}")
                                
                                # Close the chunk clip to free memory
                                chunk_clip.close()

                                # Upload chunk
                                uploaded_chunk_file = await asyncio.to_thread(genai.upload_file, chunk_path, mime_type="audio/wav")
                                logger.info(f"Uploaded chunk {i+1}: {uploaded_chunk_file.name}")

                                # Generate content for chunk (no continuation needed here)
                                chunk_text = await self._process_audio_chunk(chunk_path)
                                
                                if chunk_text:
                                    logger.info(f"Received transcript for chunk {i+1} (length: {len(chunk_text)} chars)")
                                    transcript_parts.append(chunk_text)
                                else:
                                    logger.warning(f"Failed to generate transcript for chunk {i+1}")

                            except Exception as chunk_err:
                                logger.error(f"Error processing chunk {i+1}: {chunk_err}")
                                # Optionally, decide whether to stop or continue with other chunks
                                # For now, we log the error and continue
                                transcript_parts.append(f"[ERROR PROCESSING CHUNK {i+1}]") 
                            finally:
                                # Cleanup chunk resources
                                if uploaded_chunk_file:
                                    try:
                                        await asyncio.to_thread(genai.delete_file, uploaded_chunk_file.name)
                                        logger.info(f"Deleted uploaded chunk file: {uploaded_chunk_file.name}")
                                    except Exception as delete_err:
                                        logger.warning(f"Failed to delete uploaded chunk file {uploaded_chunk_file.name}: {delete_err}")
                                if chunk_path.exists():
                                    try:
                                        os.remove(chunk_path)
                                        logger.info(f"Deleted local chunk file: {chunk_path}")
                                    except OSError as os_err:
                                        logger.warning(f"Failed to delete local chunk file {chunk_path}: {os_err}")
                        
                        # Combine results from all chunks
                        raw_text = ' '.join(transcript_parts)
                        logger.info(f"Combined transcripts from {len(transcript_parts)} chunks. Total length: {len(raw_text)} chars.")

                    except Exception as e:
                        logger.error(f"Error during audio chunking process: {e}")
                        raw_text = None # Indicate failure
                    finally:
                        # Close the full audio clip
                        if 'full_audio_clip' in locals():
                            try:
                                full_audio_clip.close()
                                logger.info("Closed full audio clip")
                            except Exception as close_err:
                                logger.warning(f"Error closing full audio clip: {close_err}")
                        
                        # Cleanup temporary directory
                        if chunk_temp_dir.exists():
                            try:
                                shutil.rmtree(chunk_temp_dir)
                                logger.info(f"Removed temporary chunk directory: {chunk_temp_dir}")
                            except OSError as rmtree_err:
                                logger.warning(f"Failed to remove temporary chunk directory {chunk_temp_dir}: {rmtree_err}")
                # --- End Audio Processing Logic ---

            except yt_dlp.utils.DownloadError as e:
                logger.error(f"yt-dlp download error during transcript generation (should not happen if audio exists): {e}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during transcript generation: {e}", exc_info=True)
                return None
            
            logger.info(f"DEBUG: After try-except: raw_text exists: {bool(raw_text)}, Length: {len(raw_text) if raw_text else 0}")
            if raw_text:
                # Save the final raw transcript from Gemini
                if audio_path: # Ensure we have a base name
                    base_filename = Path(audio_path).stem
                    raw_transcript_filename = f"{base_filename}_gemini_raw.txt"
                    raw_transcript_path = self.raw_transcript_dir / raw_transcript_filename
                    logger.info(f"Preparing to save raw transcript with length {len(raw_text)} to {raw_transcript_path}")
                    logger.info(f"Raw transcript directory exists: {self.raw_transcript_dir.exists()}, is directory: {self.raw_transcript_dir.is_dir()}")
                    
                    # Ensure raw transcript directory exists
                    os.makedirs(self.raw_transcript_dir, exist_ok=True)
                    
                    try:
                        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
                            f.write(raw_text)
                        logger.info(f"Successfully saved raw Gemini transcript to {raw_transcript_path}")
                        
                        # Verify file was created
                        if raw_transcript_path.exists():
                            logger.info(f"Verified: file exists at {raw_transcript_path} with size {raw_transcript_path.stat().st_size} bytes")
                        else:
                            logger.error(f"File verification failed - file does not exist at {raw_transcript_path} after writing")
                    except IOError as io_err:
                        logger.error(f"Failed to save raw Gemini transcript: {io_err}")
                else:
                    logger.warning("Could not save raw Gemini transcript as audio_path was missing.")
            
            logger.info(f"DEBUG: Preparing to process raw transcript into segments with {'raw_text available' if raw_text else 'NO raw_text'}")
            # Process raw transcript into segments
            words = raw_text.split()
            words_per_segment = 20
            segments = []
            total_words = len(words)
            logger.info(f"DEBUG: Processing transcript with {total_words} words")
            segment_duration = 10  # Assume 10 seconds per segment
            
            for i in range(0, total_words, words_per_segment):
                segment_words = words[i:i + words_per_segment]
                start_time = float(i / total_words * segment_duration)
                end_time = min(start_time + segment_duration, segment_duration)
                segments.append(TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=' '.join(segment_words),
                    language='en'  # Default to English for generated transcripts
                ))
            
            # Save structured transcript
            transcript_data = {
                "language": "en",
                "segments": [{
                    "start": segment.start,
                    "duration": segment.end - segment.start,
                    "text": segment.text
                } for segment in segments]
            }
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2)
            logger.info(f"Saved structured transcript to {transcript_path}")
            
            logger.info(f"DEBUG: Returning {len(segments)} transcript segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to generate transcript: {str(e)}")
            logger.info(f"DEBUG: Exception in outer try-except: {str(e)}")
            raise VideoProcessingError(f"Failed to generate transcript: {str(e)}")


   
    async def _load_raw_transcript(self, video_id: str) -> Optional[str]:
        """Load transcript text from structured transcript file with timestamps"""
        try:
            transcript_path = get_storage_path(f"videos/transcripts/{video_id}.json")
            if not transcript_path.exists():
                logger.warning(f"Transcript file not found: {transcript_path}")
                return None
                
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
                # Format each segment with its timestamp
                segments = []
                for segment in transcript_data['segments']:
                    start_time = segment['start']
                    text = segment['text']
                    segments.append(f"[{start_time:.2f}s] {text}")
                return '\n'.join(segments)
                
        except Exception as e:
            logger.error(f"Failed to load transcript: {str(e)}")
            return None

    def _save_raw_transcript(self, video_id: str, text: str):
        """Save raw transcript text to file"""
        try:
            file_path = get_storage_path(f"videos/raw_transcripts/{video_id}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved raw transcript to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save raw transcript: {str(e)}")
            raise VideoProcessingError(f"Failed to save raw transcript: {str(e)}")
            
    async def _extract_frames(self, video: Video) -> List[str]:
        """Extract frames from video at regular intervals"""
        try:
            logger.info(f"Extracting frames from video {video.id}")
            
            # Create video-specific frames directory
            frames_dir = get_storage_subdir(f"videos/frames/{video.id}")
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Open video file
            video_clip = VideoFileClip(video.file_path)
            duration = video_clip.duration
            frame_interval = 5  # Extract a frame every 5 seconds
            
            # Calculate total frames to extract
            total_frames = int(duration / frame_interval)
            frames_extracted = 0
            frame_paths = []
            
            logger.info(f"Will extract approximately {total_frames} frames")
            
            try:
                # Extract frames at intervals
                for time in range(0, int(duration), frame_interval):
                    # Get frame at current time
                    frame = video_clip.get_frame(time)
                    
                    # Convert to PIL Image and resize if needed
                    from PIL import Image
                    import numpy as np
                    frame_pil = Image.fromarray(np.uint8(frame))
                    
                    # Save frame as image
                    frame_path = str(frames_dir / f"frame_{time:04d}.jpg")
                    frame_pil.save(frame_path, quality=90)
                    
                    frame_paths.append(frame_path)
                    frames_extracted += 1
                    
                    # Update progress
                    if total_frames > 0:
                        progress = (frames_extracted / total_frames) * 100
                        self.video_status[video.id]["progress"]["extracting_frames"] = progress
                        logger.debug(f"Frame extraction progress: {progress:.1f}%")
                
                logger.info(f"Extracted {frames_extracted} frames from video {video.id}")
                return frame_paths
                
            finally:
                # Clean up
                video_clip.close()
                
        except Exception as e:
            logger.error(f"Failed to extract frames from video {video.id}: {str(e)}", exc_info=True)
            return []

    def _save_transcript_locally(self, video_id: str, transcript: List[TranscriptSegment]) -> None:
        """Save transcript to local file"""
        try:
            # Save segmented transcript as JSON
            transcript_path = get_storage_path(f"videos/transcripts/{video_id}.json")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                # Convert TranscriptSegment objects to dict, excluding any None values
                segments = []
                for segment in transcript:
                    segment_dict = segment.dict(exclude_none=True)
                    segments.append(segment_dict)
                json.dump(segments, f, indent=2)
            logger.info(f"Saved segmented transcript to {transcript_path}")
            
        except Exception as e:
            logger.error(f"Failed to save transcript locally: {str(e)}")
            raise VideoProcessingError(f"Failed to save transcript: {str(e)}")

    async def _align_frames_with_transcript(self, video: Video, frames: List[str]) -> List[TranscriptSegment]:
        """Align frames with transcript segments."""
        try:
            if not video.analysis or not video.analysis.transcript:
                logger.warning("No transcript available for frame alignment")
                return []

            logger.info("Aligning frames with transcript segments")
            segments = []
            # Implementation of frame-transcript alignment
            return segments

        except Exception as e:
            logger.error(f"Failed to align frames with transcript: {str(e)}")
            raise VideoProcessingError(f"Failed to align frames with transcript: {str(e)}")

    async def _save_analysis_json(self, video: Video):
        """Save analysis results to JSON file."""
        try:
            analysis_path = get_storage_path(f"videos/analysis/{video.id}.json")
            logger.info(f"Saving analysis to {analysis_path}")
            
            # Create analysis directory if it doesn't exist
            analysis_dir = get_storage_subdir(f"videos/analysis")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert analysis to dict and save
            if video.analysis:
                analysis_dict = video.analysis.model_dump()
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_dict, f, indent=4)
                logger.info("Analysis JSON saved successfully")
            else:
                logger.warning("No analysis available to save")
        except Exception as e:
            logger.error(f"Failed to save analysis JSON: {str(e)}")
            raise VideoProcessingError(f"Failed to save analysis JSON: {str(e)}")

    async def _analyze_content(self, video: Video) -> Video:
        """Analyze video content using the content analyzer."""
        try:
            logger.info(f"Starting content analysis for video {video.id}")
            
            # Load raw transcript
            raw_transcript = await self._load_raw_transcript(video.id)
            if not raw_transcript:
                raise VideoProcessingError("No transcript available for analysis")
            
            # Process transcript to identify stock discussions and segments
            transcript_analysis = await self.content_analyzer._process_transcript(video, raw_transcript)
            
            if not transcript_analysis or 'analysis_json' not in transcript_analysis:
                raise VideoProcessingError("Failed to analyze transcript content")
            
            # Create initial analysis object
            video.analysis = VideoAnalysis(
                transcript=raw_transcript,
                summary=transcript_analysis.get('raw_response', ''),
                key_frames=video.analysis.key_frames if video.analysis and video.analysis.key_frames else [],
                segments=transcript_analysis.get('stocks', []),
                frame_analysis={},  # Will be updated after frame analysis
                embedding_id=""  # Will be set when embedding is created
            )
            
            # Save initial analysis for frame processing
            initial_analysis_path = self.content_analyzer.summaries_dir / f"{video.id}_initial_analysis.json"
            with open(initial_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_analysis['analysis_json'], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved initial analysis to {initial_analysis_path}")

            # Generate frames report and analyze frames
            try:
                # First ensure we have frames to analyze
                if not video.analysis.key_frames:
                    logger.warning("No frames available for analysis")
                    return video

                logger.info(f"Generating frames report for video {video.id}")
                report_path = self.content_analyzer.report_generator.generate_frames_report(video)
                if not report_path:
                    raise VideoProcessingError("Failed to generate frames report")
                
                # Ensure report was generated
                report_path = Path(report_path)
                if not report_path.exists():
                    raise VideoProcessingError(f"Frames report not found at {report_path}")
                
                logger.info(f"Generated frames report at {report_path}")
                
                # Analyze frames and update summary
                logger.info("Analyzing frames and updating summary...")
                updated_analysis = await self.content_analyzer._analyze_frames_and_update_summary(
                    video=video,
                    frames_pdf_path=str(report_path),
                    summary_json_path=str(initial_analysis_path)
                )
            
                # Update video analysis with frame information
                if updated_analysis and 'sections' in updated_analysis:
                    video.analysis.frame_analysis = updated_analysis
                    logger.info("Successfully updated analysis with frame information")
                    
                    # Save the final analysis
                    final_output_path = self.content_analyzer.summaries_dir / f"{video.id}_final_analysis.json"
                    logger.info(f"Saving final analysis to {final_output_path}")
                    
                    # Clean the updated_analysis before saving
                    # This ensures all keys and string values are properly formatted for JSON
                    def clean_for_json(obj):
                        if isinstance(obj, dict):
                            # Create a new dict with properly handled values
                            cleaned_dict = {}
                            for k, v in obj.items():
                                # Handle the 'sections' field specifically if it's a string containing JSON
                                if k == 'sections' and isinstance(v, str):
                                    try:
                                        # Try to parse the sections string as JSON
                                        logger.info("Parsing 'sections' field from string to JSON object")
                                        cleaned_dict[str(k)] = json.loads(v)
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Failed to parse 'sections' as JSON: {str(e)}")
                                        # Keep as string if parsing fails
                                        cleaned_dict[str(k)] = v
                                else:
                                    cleaned_dict[str(k)] = clean_for_json(v)
                            return cleaned_dict
                        elif isinstance(obj, list):
                            return [clean_for_json(i) for i in obj]
                        elif isinstance(obj, (str, int, float, bool)) or obj is None:
                            return obj
                        else:
                            # Convert any other types to strings
                            return str(obj)
                    
                    # Clean the analysis object before saving
                    cleaned_analysis = clean_for_json(updated_analysis)
                
                    try:
                        with open(final_output_path, 'w', encoding='utf-8') as f:
                            json.dump(cleaned_analysis, f, indent=2, ensure_ascii=False)
                        logger.info(f"Saved final analysis to {final_output_path}")
                    except Exception as e:
                        logger.error(f"Error saving final analysis JSON: {str(e)}")
                        # Fallback: try to save with more aggressive sanitization
                        try:
                            # Convert to string, clean, and parse back to ensure valid JSON
                            analysis_str = json.dumps(cleaned_analysis)
                            # Remove any potential problematic characters
                            analysis_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', analysis_str)
                            # Try to parse back to verify
                            reparsed = json.loads(analysis_str)
                            # Save the sanitized version
                            with open(final_output_path, 'w', encoding='utf-8') as f:
                                json.dump(reparsed, f, indent=2, ensure_ascii=False)
                            logger.info(f"Saved sanitized final analysis to {final_output_path}")
                        except Exception as fallback_error:
                            logger.error(f"Failed even with sanitization: {str(fallback_error)}")
            
            except Exception as e:
                logger.error(f"Failed to analyze frames: {str(e)}", exc_info=True)
                logger.warning("Continuing with original analysis without frames")
            
            # Agent workflow has been moved to the main process_video method
            # to ensure it runs after both content analysis and frame extraction
            
            # Update video status
            self.video_status[video.id]["status"] = "completed"
            logger.info(f"Video {video.id} processed successfully")

            return video
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {str(e)}")
            raise VideoProcessingError(f"Failed to analyze content: {str(e)}")
            
    async def _trigger_agent_workflow(self, video: Video) -> Dict:
        """
        Trigger the agent workflow with the video analysis results.
        
        Args:
            video: Video object with analysis data
            
        Returns:
            Dict containing agent processing results
        """
        try:
            # First, make sure we have a valid video object
            if not video or not hasattr(video, 'id'):
                logger.error("Invalid video object passed to agent workflow")
                return {"status": "error", "error": "Invalid video object"}
                
            logger.info(f"Preparing to trigger AgentWorkflow for video {video.id}")
            
            # Safely import AgentWorkflow from agents module
            try:
                from app.agents.agents import AgentWorkflow
            except ImportError as e:
                logger.error(f"Failed to import AgentWorkflow: {str(e)}")
                return {"status": "error", "error": f"Failed to import AgentWorkflow: {str(e)}"}
            
            # Initialize AgentWorkflow - configuration loading is handled internally
            try:
                workflow = AgentWorkflow()
                logger.info("AgentWorkflow initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AgentWorkflow: {str(e)}", exc_info=True)
                return {"status": "error", "error": f"Failed to initialize AgentWorkflow: {str(e)}"}
            
            # Process the video analysis
            try:
                # Prepare the data in the format expected by the AgentWorkflow
                if hasattr(video, 'analysis') and video.analysis:
                    # Load the final analysis and prepare it for AgentWorkflow
                    logger.info(f"Preparing analysis data for video {video.id}")
                    
                    # Get the raw analysis data either from the file or the object
                    analysis_data = {}
                    
                    try:
                        # Try to get from JSON file first
                        storage_base = config.storage.base_path if hasattr(config, 'storage') and hasattr(config.storage, 'base_path') else '.'
                        summaries_dir = Path(storage_base) / 'videos' / 'summaries'
                        final_analysis_path = summaries_dir / f"{video.id}_final_analysis.json"
                        
                        if final_analysis_path.exists():
                            with open(final_analysis_path, 'r') as f:
                                analysis_data = json.load(f)
                                # Explicitly add the video ID
                                analysis_data['video_id'] = video.id
                                logger.info(f"Loaded analysis data from file for video {video.id}")
                        else:
                            # Use the analysis object directly
                            analysis_data = video.analysis.dict() if hasattr(video.analysis, 'dict') else vars(video.analysis)
                            # Explicitly add the video ID
                            analysis_data['video_id'] = video.id
                            logger.info(f"Using in-memory analysis data for video {video.id}")
                    except Exception as data_error:
                        logger.error(f"Error loading analysis data: {str(data_error)}", exc_info=True)
                        analysis_data = {"video_id": video.id, "error": "Failed to load analysis data"}
                        
                    # Execute the workflow
                    logger.info(f"Executing agent workflow for video {video.id}")
                    results = await workflow.execute(analysis_data)
                    logger.info(f"Agent workflow completed for video: {video.id}")
                    return results
                else:
                    logger.error(f"No analysis data available for video {video.id}")
                    return {"status": "error", "error": "No analysis data available"}
            except Exception as e:
                logger.error(f"Error in agent workflow execution for video {video.id}: {str(e)}", exc_info=True)
                return {"status": "error", "error": f"Error in workflow execution: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Unexpected error in agent workflow for video: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}
