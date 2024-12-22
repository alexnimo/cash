import asyncio
from typing import Dict, List, Optional, Any
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
import google.generativeai as genai
from app.models.video import Video, VideoSource, VideoStatus, VideoMetadata, VideoAnalysis
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
                raw_transcript_path = self.raw_transcript_dir / f"{video.id}.txt"
                with open(raw_transcript_path, 'w', encoding='utf-8') as f:
                    f.write(raw_text)
                logger.info(f"Saved raw transcript to {raw_transcript_path}")
                
                # Save structured transcript to file
                transcript_path = self.transcript_dir / f"{video.id}.json"
                logger.info(f"Saving transcript to {transcript_path}")
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "language": transcript_language,
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

            # Extract metadata
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

            # Generate frames report after all processing is complete
            try:
                logger.info(f"Generating frames report for video {video.id}")
                report_path = self.content_analyzer.report_generator.generate_frames_report(video)
                logger.info(f"Successfully generated frames report at {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate frames report: {str(e)}", exc_info=True)

            # Update status
            self.video_status[video.id]["status"] = "completed"
            logger.info(f"Video {video.id} processed successfully")

            return video

        except Exception as e:
            logger.error(f"Failed to process video: {str(e)}")
            if video.id in self.video_status:
                self.video_status[video.id]["status"] = "failed"
            raise VideoProcessingError(f"Failed to process video: {str(e)}")

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
            
            if not transcript_analysis or 'raw_response' not in transcript_analysis:
                raise VideoProcessingError("Failed to analyze transcript content")
            
            # Create final analysis object
            video.analysis = VideoAnalysis(
                transcript=raw_transcript,
                summary=transcript_analysis.get('raw_response', ''),
                key_frames=video.analysis.key_frames if video.analysis and video.analysis.key_frames else [],
                segments=transcript_analysis.get('stocks', []),
                frame_analysis={},  # Empty for now
                embedding_id=""  # Will be set when embedding is created
            )
            
            # Update video status
            self.video_status[video.id]["status"] = VideoStatus.COMPLETED
            self.video_status[video.id]["progress"]["analysis"] = 100
            
            logger.info("Content analysis completed successfully")
            return video
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {str(e)}")
            raise VideoProcessingError(f"Failed to analyze content: {str(e)}")

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
        try:
            video_url = str(video.url)
            logger.info(f"Downloading video from URL: {video_url}")
            
            # Download video only
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
            audio_path = str(self.audio_dir / f'{video_id}.wav')
            
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

    async def _generate_transcript(self, audio_path: str) -> List[TranscriptSegment]:
        """Generate transcript from audio file using Gemini"""
        try:
            logger.info(f"Generating transcript for audio {audio_path}")
            
            # Check if transcript exists first
            video_id = Path(audio_path).stem
            transcript_path = self.transcript_dir / f"{video_id}.json"
            if transcript_path.exists():
                logger.info(f"Found existing transcript at {transcript_path}")
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    return [TranscriptSegment(
                        start_time=segment['start'],
                        end_time=segment['start'] + segment['duration'],
                        text=segment['text'],
                        language=transcript_data['language']
                    ) for segment in transcript_data['segments']]
                    
            # If no transcript exists, generate one
            try:
                # Add exponential backoff retry logic
                base_delay = 1  # Start with 1 second delay
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Single Gemini API call to get transcript
                        raw_text = await self.model_manager.transcribe_audio(audio_path)
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
            transcript_path = self.transcript_dir / f"{video_id}.json"
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
            file_path = self.raw_transcript_dir / f"{video_id}.txt"
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

    async def _save_analysis_json(self, video_id: str, analysis: Dict[str, Any]) -> None:
        """Save the analysis in JSON format"""
        try:
            analysis_path = self.transcript_dir / f"{video_id}_analysis.json"
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved analysis JSON to {analysis_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis JSON: {str(e)}")
            raise VideoProcessingError(f"Failed to save analysis JSON: {str(e)}")
