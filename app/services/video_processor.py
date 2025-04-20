import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
import os
from pathlib import Path
import logging
from moviepy import VideoFileClip, AudioFileClip
import yt_dlp
import json
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from app.utils.langtrace_utils import get_langtrace, init_langtrace
from app.utils.path_utils import get_storage_subdir, get_storage_path
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from app.models.video import Video, VideoSource, VideoStatus, VideoMetadata, VideoAnalysis
from app.core.settings import get_settings
from app.services.model_manager import ModelManager
from app.services.model_config import configure_models
from app.services.content_analyzer import ContentAnalyzer
from app.services.errors import VideoProcessingError
from app.services.json_utils import lint_json, fix_json, clean_json_response
import re

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
settings = get_settings()

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Initialize LangTrace at module level
_tracer = init_langtrace()
if _tracer:
    logger.info("LangTrace initialized successfully at module level")
else:
    logger.warning("Failed to initialize LangTrace at module level")

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
    
    # Before returning, try to validate and fix the JSON if needed
    if is_json:
        try:
            # Try to parse as JSON first
            try:
                json.loads(combined_text)
                logger.info("Final chunked response is valid JSON, no fixes needed")
            except json.JSONDecodeError:
                # If parsing fails, use our linter and fixer
                logger.info("Final chunked response appears to be JSON but has issues, attempting to fix...")
                error_info = lint_json(combined_text)
                
                if error_info:
                    logger.warning(f"JSON issues in final combined response: {error_info['readable_error']}")
                    fixed_json, fixes_applied = fix_json(combined_text)
                    
                    if fixed_json:
                        logger.info(f"Successfully fixed JSON issues in chunked response. Fixes: {len(fixes_applied)}")
                        # Verify the fixed JSON is valid
                        try:
                            json.loads(fixed_json)
                            combined_text = fixed_json  # Use the fixed version
                            logger.info("Using fixed JSON for final response")
                        except json.JSONDecodeError as e:
                            logger.error(f"Fixed JSON still invalid: {str(e)}")
                    else:
                        logger.warning(f"Could not fix JSON issues in chunked response: {fixes_applied}")
        except Exception as e:
            logger.warning(f"Error validating final chunked response as JSON: {str(e)}")
    
    return combined_text

class VideoProcessor:
    def __init__(self, video_status: Dict[str, Dict]):
        try:
            # Use the unified storage path utilities
            self.video_dir = get_storage_subdir("videos")
            self.audio_dir = get_storage_subdir("videos/audio")
            self.transcript_dir = get_storage_subdir("videos/transcripts")
            self.raw_transcript_dir = get_storage_subdir("videos/raw_transcripts")
            self.frames_dir = get_storage_subdir("videos/frames")
            
            # Create trace directory if langtrace is enabled
            if settings.langtrace.enabled:
                trace_dir = Path(settings.langtrace.trace_dir)
                if not trace_dir.is_absolute():
                    trace_dir = get_storage_path(settings.langtrace.trace_dir)
                    trace_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize services
            self.model_manager = ModelManager(settings)
            self.content_analyzer = ContentAnalyzer()
            
            # Initialize semaphore for parallel processing
            self.semaphore = asyncio.Semaphore(settings.processing.max_parallel_chunks)
            
            # Store reference to global video status
            self.video_status = video_status
            
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

            # Get available transcripts
            logger.info("Fetching available transcripts...")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Log available transcripts
            available_transcripts = list(transcript_list)
            logger.info(f"Found {len(available_transcripts)} available transcripts:")
            for t in available_transcripts:
                logger.info(f"- Language: {t.language_code}, Generated: {t.is_generated}")
            
            transcript = None
            transcript_language = None
            
            try:
                # First try to get English transcript (manual or generated)
                logger.info("Trying to get English transcript...")
                transcript = transcript_list.find_transcript(['en'])
                transcript_language = 'en'
                logger.info("Found English transcript")
            except NoTranscriptFound:
                logger.info("No English transcript found, trying any available transcript...")
                # Get all available transcripts
                available = list(transcript_list)
                if available:
                    # Use the first available transcript
                    transcript = available[0]
                    transcript_language = transcript.language_code
                    logger.info(f"Using available transcript in language: {transcript_language}")
                else:
                    logger.warning("No transcripts available")
                    return None

            if transcript:
                # Fetch the transcript data
                logger.info(f"Fetching transcript data in {transcript_language}...")
                transcript_data = transcript.fetch()
                logger.info(f"Got transcript with {len(transcript_data)} segments")
                
                # Save raw transcript text
                raw_text = ' '.join(segment['text'] for segment in transcript_data)
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
                        "segments": transcript_data
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

    async def process_video(self, video: Video) -> Video:
        """Process a video by downloading it and generating a transcript."""
        try:
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

            # Download video
            logger.info(f"Processing video {video.id}")
            video_path = await self._download_video(video)
            video.file_path = video_path
            logger.info(f"Video downloaded to {video_path}")

            # Extract frames
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
            else:
                logger.warning("No frames were extracted")

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
                    transcript_path = await self._get_youtube_transcript(video)
                    if transcript_path:
                        logger.info(f"Successfully retrieved YouTube transcript from {transcript_path}")
                        # Parse the saved transcript file
                        with open(transcript_path) as f:
                            transcript_data = json.load(f)
                            logger.info(f"Loaded transcript data with language: {transcript_data.get('language')}")
                            video.transcript = [TranscriptSegment(
                                start_time=entry['start'],
                                end_time=entry['start'] + entry['duration'],
                                text=entry['text'],
                                language=transcript_data["language"]
                            ) for entry in transcript_data["segments"]]
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
                except Exception as e:
                    logger.error(f"Failed to generate transcript: {str(e)}")
                    raise VideoProcessingError(f"Failed to generate transcript: {str(e)}")

            # Analyze content
            logger.info(f"Analyzing content for video {video.id}")
            video = await self._analyze_content(video)
            self.video_status[video.id]["progress"]["analyzing"] = 100
            logger.info("Content analysis completed")

            # Update status
            self.video_status[video.id]["status"] = "completed"
            logger.info(f"Video {video.id} processed successfully")

            return video

        except Exception as e:
            logger.error(f"Failed to process video: {str(e)}")
            if video.id in self.video_status:
                self.video_status[video.id]["status"] = "failed"
            raise VideoProcessingError(f"Failed to process video: {str(e)}")

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
            
            # First extract metadata without downloading
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
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
            
            # Download video only
            video_opts = {
                'format': 'bestvideo[height<=?1080]/mp4',  # Get best video up to 1080p
                'outtmpl': str(self.video_dir / f'{video.id}.%(ext)s'),
                'progress_hooks': [lambda d: self._update_download_progress(video.id, d)],
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': False,
                'nocheckcertificate': True,
                'source_address': '0.0.0.0',  # Force ipv4  # Use cookies file
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
            logger.error(f"Failed to download video: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to download video: {str(e)}")

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
                for attempt in range(max_retries):
                    try:
                        # Get the Gemini model response with chunking support
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
                        
                        # Upload audio file
                        audio_file = genai.upload_file(audio_path, mime_type="audio/wav")
                        logger.info(f"Successfully uploaded audio file: {audio_path}")
                        
                        # Generate initial response
                        initial_response = await asyncio.to_thread(
                            model.generate_content, 
                            contents=[transcription_prompt, audio_file],
                            request_options={"timeout": 1000}
                        )
                        
                        # Check if response is truncated and handle chunking if needed
                        if is_response_truncated(initial_response):
                            logger.info("Initial transcription response was truncated, using chunking mechanism")
                            # Use chunking for potentially long responses
                            raw_text = await handle_chunked_response(
                                model=model,
                                original_prompt=transcription_prompt,
                                first_response=initial_response,
                                additional_contents=[audio_file],
                                max_continuation_attempts=5,
                                is_json=False  # Raw transcript is not JSON
                            )
                        else:
                            # Response was not truncated, use it directly
                            raw_text = initial_response.text.strip()
                        
                        if not raw_text:
                            raise VideoProcessingError("Received empty transcript from model")
                        
                        # Save raw transcript immediately after successful generation
                        self._save_raw_transcript(Path(audio_path).stem, raw_text)
                        break  # Success! Break out of retry loop
                        
                    except Exception as e:
                        if "429" in str(e) or "Resource has been exhausted" in str(e):
                            if attempt < max_retries - 1:  # Don't wait on last attempt
                                wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                                await asyncio.sleep(wait_time)
                                continue
                        raise  # Re-raise if not a rate limit error or out of retries
                else:
                    raise VideoProcessingError("Failed to transcribe after maximum retries")
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Resource has been exhausted" in error_msg:
                    error_msg = "API rate limit exceeded. Please try again later."
                logger.error(f"Transcription failed: {error_msg}")
                raise VideoProcessingError(f"Failed to generate transcript: {error_msg}")
            
            # Process raw transcript into segments
            words = raw_text.split()
            words_per_segment = 20
            segments = []
            total_words = len(words)
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
                    "start": segment.start_time,
                    "duration": segment.end_time - segment.start_time,
                    "text": segment.text
                } for segment in segments]
            }
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2)
            logger.info(f"Saved structured transcript to {transcript_path}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to generate transcript: {str(e)}")
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

    async def _extract_frames(self, video: Video) -> List[str]:
        """Extract frames from video at 5-second intervals"""
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
                    
                    # Resize frame to 1920x1080
                    from PIL import Image
                    import numpy as np
                    frame_pil = Image.fromarray(np.uint8(frame))
                    frame_pil = frame_pil.resize((1920, 1080), Image.Resampling.LANCZOS)
                    
                    # Save frame as image
                    frame_path = str(frames_dir / f"frame_{time:04d}.jpg")
                    frame_pil.save(frame_path, quality=95)  # High quality JPEG
                    
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
            raise VideoProcessingError(f"Failed to extract frames: {str(e)}")

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
            
            # Update video status
            self.video_status[video.id]["status"] = "completed"
            logger.info(f"Video {video.id} processed successfully")

            return video
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {str(e)}")
            raise VideoProcessingError(f"Failed to analyze content: {str(e)}")
