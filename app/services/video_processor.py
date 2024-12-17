import asyncio
from typing import Dict, List, Optional
import os
from pathlib import Path
import logging
from moviepy import VideoFileClip, AudioFileClip
import yt_dlp
import json
from datetime import datetime
from app.utils.langtrace_utils import get_langtrace, init_langtrace
import google.generativeai as genai
from app.models.video import Video, VideoSource, VideoStatus, VideoMetadata
from app.models.transcript import TranscriptSegment
from app.core.settings import get_settings
from app.services.model_manager import ModelManager
from app.services.model_config import configure_models
from app.services.content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize LangTrace at module level
_tracer = init_langtrace()
if _tracer:
    logger.info("LangTrace initialized successfully at module level")
else:
    logger.warning("Failed to initialize LangTrace at module level")

# Configure models
configure_models()

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.absolute()

class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class VideoProcessor:
    def __init__(self, video_status: Dict[str, Dict]):
        try:
            # Convert relative paths to absolute paths using Path for OS agnostic handling
            base_dir = Path(settings.video_storage.base_dir)
            if not base_dir.is_absolute():
                base_dir = PROJECT_ROOT / base_dir
            self.video_dir = base_dir.resolve()
            self.video_dir.mkdir(parents=True, exist_ok=True)
            
            # Create audio directory for transcription
            self.audio_dir = (self.video_dir / "audio").resolve()
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Create transcripts directory
            self.transcript_dir = (self.video_dir / "transcripts").resolve()
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            
            # Create raw transcripts directory for text files
            self.raw_transcript_dir = (self.video_dir / "raw_transcripts").resolve()
            self.raw_transcript_dir.mkdir(parents=True, exist_ok=True)
            
            # Create frames directory
            self.frames_dir = (self.video_dir / "frames").resolve()
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Create trace directory if langtrace is enabled
            if settings.langtrace.enabled:
                trace_dir = Path(settings.langtrace.trace_dir)
                if not trace_dir.is_absolute():
                    trace_dir = (PROJECT_ROOT / trace_dir).resolve()
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

    async def process_video(self, video: Video) -> Video:
        """Process a video for analysis"""
        try:
            video.status = VideoStatus.PROCESSING
            self.video_status[video.id].update({
                "status": VideoStatus.PROCESSING,
                "progress": {
                    "downloading": 0,
                    "transcribing": 0,
                    "analyzing": 0,
                    "extracting_frames": 0
                }
            })
            logger.info(f"Processing video {video.id}")
            
            # Download YouTube video
            logger.info("Starting video download")
            video.file_path = await self._download_video(video)
            self.video_status[video.id]["progress"]["downloading"] = 100
            logger.info(f"Downloaded video to {video.file_path}")
            
            # Extract metadata
            video.metadata = await self._extract_metadata(video.file_path)
            logger.info(f"Extracted metadata: duration={video.metadata.duration}s")
            
            # Extract frames
            logger.info("Starting frame extraction")
            self.video_status[video.id]["progress"]["extracting_frames"] = 0
            frames = await self._extract_frames(video)
            self.video_status[video.id]["progress"]["extracting_frames"] = 100
            logger.info("Frame extraction completed")
            
            # Initialize transcription progress
            self.video_status[video.id]["progress"]["transcribing"] = 0
            
            # Check if transcript exists locally first
            transcript_path = self.transcript_dir / f"{video.id}.json"
            if transcript_path.exists():
                logger.info(f"Found local transcript at {transcript_path}")
                with open(transcript_path) as f:
                    video.transcript = [TranscriptSegment(**segment) for segment in json.load(f)]
                self.video_status[video.id]["progress"]["transcribing"] = 100
            else:
                # Extract audio and generate transcript
                video.audio_path = await self._extract_audio(video.file_path)
                video.transcript = await self._generate_transcript(video)
                
                # Save transcript locally
                self._save_transcript_locally(video.id, video.transcript)
                self.video_status[video.id]["progress"]["transcribing"] = 100
                logger.info("Transcript generation and storage completed")

            # Align frames with transcript segments
            logger.info("Aligning frames with transcript segments")
            video.transcript = await self._align_frames_with_transcript(video, frames)
            
            # Analyze content and generate summaries
            logger.info("Analyzing video content")
            self.video_status[video.id]["progress"]["analyzing"] = 0
            summaries = await self.content_analyzer.analyze_video_content(video)
            self.video_status[video.id]["progress"]["analyzing"] = 100
            logger.info(f"Generated {len(summaries)} content summaries")
            
            # Mark as completed
            video.status = VideoStatus.COMPLETED
            self.video_status[video.id]["status"] = VideoStatus.COMPLETED
            logger.info(f"Completed processing video {video.id}")
            return video
            
        except Exception as e:
            video.status = VideoStatus.FAILED
            video.error = str(e)
            self.video_status[video.id].update({
                "status": VideoStatus.FAILED,
                "error": str(e)
            })
            logger.error(f"Failed to process video: {str(e)}")
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
        # First download video only
        video_opts = {
            'format': 'bestvideo[height<=?720]/mp4',  # Get best video up to 720p
            'outtmpl': str(self.video_dir / f'{video.id}.%(ext)s'),
            'progress_hooks': [lambda d: self._update_download_progress(video.id, d)],
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'geo_bypass': True
        }
        
        try:
            video_url = str(video.url)
            logger.info(f"Downloading video from URL: {video_url}")
            
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
                
                # Download audio separately
                audio_opts = {
                    'format': 'bestaudio/wav',  # Get best audio
                    'outtmpl': str(self.audio_dir / f'{video.id}.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'ignoreerrors': False,
                    'nocheckcertificate': True,
                    'geo_bypass': True
                }
                
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    if info is None:
                        raise VideoProcessingError("Failed to download audio")
                    
                    # Store the audio path in the video object
                    audio_path = str(self.audio_dir / f'{video.id}.wav')
                    if not os.path.exists(audio_path):
                        potential_audio_files = list(self.audio_dir.glob(f'{video.id}.*'))
                        if not potential_audio_files:
                            raise VideoProcessingError(f"Downloaded audio file not found for ID: {video.id}")
                        audio_path = str(potential_audio_files[0])
                    
                    video.audio_path = audio_path
                    logger.info(f"Successfully downloaded audio to: {audio_path}")
                
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
        """Return path to the already downloaded audio file"""
        try:
            video_id = Path(video_path).stem
            audio_path = str(self.audio_dir / f'{video_id}.wav')
            
            # Check for audio file with any extension if exact path not found
            if not os.path.exists(audio_path):
                potential_files = list(self.audio_dir.glob(f'{video_id}.*'))
                if not potential_files:
                    raise VideoProcessingError("Audio file not found")
                audio_path = str(potential_files[0])
            
            logger.info(f"Using audio file: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to get audio path: {str(e)}")
            raise VideoProcessingError(f"Failed to extract audio: {str(e)}")

    async def _extract_metadata(self, video_path: str) -> VideoMetadata:
        """Extract metadata from video file"""
        try:
            logger.info(f"Extracting metadata from video: {video_path}")
            video = VideoFileClip(video_path)
            
            metadata = VideoMetadata(
                title=Path(video_path).stem,
                duration=video.duration,
                resolution=(video.size[0], video.size[1]),
                format=Path(video_path).suffix[1:],
                size_bytes=os.path.getsize(video_path)
            )
            
            video.close()
            logger.info(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to extract metadata: {str(e)}")

    async def _generate_transcript(self, video: Video) -> List[TranscriptSegment]:
        """Generate transcript from audio file using Gemini"""
        try:
            logger.info(f"Generating transcript for video {video.id}")
            
            # Check if raw transcript exists first
            raw_text = self._load_raw_transcript(video.id)
            if raw_text:
                logger.info(f"Found existing raw transcript for video {video.id}")
            else:
                try:
                    # Add exponential backoff retry logic
                    max_retries = 5
                    base_delay = 2  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            # Single Gemini API call to get transcript
                            raw_text = await self.model_manager.transcribe_audio(video.audio_path)
                            if not raw_text:
                                raise VideoProcessingError("Received empty transcript from model")
                            
                            # Save raw transcript immediately after successful generation
                            self._save_raw_transcript(video.id, raw_text)
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
            
            # Validate transcript text
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise VideoProcessingError("Invalid transcript received from model")
            
            # Simple segmentation without using Gemini
            words = raw_text.split()
            if not words:
                raise VideoProcessingError("Transcript contains no words")
            
            words_per_segment = 20
            segments = []
            total_words = len(words)
            segment_duration = video.metadata.duration / (total_words / words_per_segment)
            
            for i in range(0, total_words, words_per_segment):
                segment_words = words[i:i + words_per_segment]
                start_time = float(i / total_words * video.metadata.duration)
                end_time = min(start_time + segment_duration, video.metadata.duration)
                segments.append(TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=" ".join(segment_words)
                ))
            
            if not segments:
                raise VideoProcessingError("Failed to generate transcript segments")
            
            logger.info(f"Generated {len(segments)} transcript segments")
            return segments
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Resource has been exhausted" in error_msg:
                error_msg = "API rate limit exceeded. Please try again later."
            logger.error(f"Failed to generate transcript: {error_msg}")
            raise VideoProcessingError(f"Failed to generate transcript: {error_msg}")

    def _save_raw_transcript(self, video_id: str, text: str) -> None:
        """Save raw transcript text to file"""
        try:
            raw_transcript_path = self.raw_transcript_dir / f"{video_id}.txt"
            with open(raw_transcript_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Saved raw transcript to {raw_transcript_path}")
        except Exception as e:
            logger.error(f"Failed to save raw transcript: {str(e)}")
            # Don't raise error since this is not critical

    def _load_raw_transcript(self, video_id: str) -> Optional[str]:
        """Load raw transcript text from file"""
        try:
            raw_transcript_path = self.raw_transcript_dir / f"{video_id}.txt"
            if raw_transcript_path.exists():
                with open(raw_transcript_path) as f:
                    return f.read()
            return None
        except Exception as e:
            logger.error(f"Failed to load raw transcript: {str(e)}")
            return None

    def _save_transcript_locally(self, video_id: str, transcript: List[TranscriptSegment]) -> None:
        """Save transcript to local file"""
        try:
            # Save segmented transcript as JSON
            transcript_path = self.transcript_dir / f"{video_id}.json"
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
            frames_dir = self.frames_dir / video.id
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
                    
                    # Resize frame to 1280x720
                    from PIL import Image
                    import numpy as np
                    frame_pil = Image.fromarray(np.uint8(frame))
                    frame_pil = frame_pil.resize((1280, 720), Image.Resampling.LANCZOS)
                    
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
        """Align extracted frames with transcript segments and save mapping"""
        if not video.transcript or not frames:
            return video.transcript

        try:
            # Sort frames by timestamp
            frame_times = []
            for frame in frames:
                # Extract timestamp from frame filename (e.g., frame_10.jpg -> 10 seconds)
                timestamp = float(frame.split('_')[1].split('.')[0])
                frame_times.append((timestamp, frame))
            frame_times.sort()  # Sort by timestamp

            # Create frame-transcript mapping
            frame_mapping = {
                "video_id": video.id,
                "mapping": []
            }
            
            # Align frames with transcript segments
            for segment in video.transcript:
                # Find all frames that fall within this segment's time range
                segment_frames = [
                    frame for time, frame in frame_times
                    if segment.start_time <= time <= segment.end_time
                ]
                
                # Add mapping entry
                frame_mapping["mapping"].append({
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "transcript": segment.text,
                    "frames": segment_frames
                })
            
            # Save frame mapping to file
            mapping_path = self.transcript_dir / f"{video.id}_frame_mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(frame_mapping, f, indent=2)
            logger.info(f"Saved frame mapping to {mapping_path}")
            
            return video.transcript
            
        except Exception as e:
            logger.error(f"Failed to align frames with transcript: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to align frames with transcript: {str(e)}")
