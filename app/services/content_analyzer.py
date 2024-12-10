import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai
from datetime import datetime, timedelta
from app.core.config import get_settings
from app.models.video import Video, VideoStatus
from app.services.model_manager import ModelManager
from app.utils.langtrace_utils import get_langtrace, trace_llm_call, init_langtrace
from PIL import Image

logger = logging.getLogger(__name__)
settings = get_settings()

class ContentAnalyzer:
    def __init__(self):
        """Initialize the ContentAnalyzer."""
        self.settings = get_settings()
        self.model_manager = ModelManager(self.settings)
        
        # Initialize model as None (it will be lazily loaded)
        self.video_analysis_model = None
        
        # Get the absolute path to the project root directory
        self.project_root = Path(__file__).resolve().parent.parent.parent.absolute()
        
        # Convert relative paths to absolute paths
        base_dir = Path(self.settings.video_storage.base_dir)
        if not base_dir.is_absolute():
            base_dir = self.project_root / base_dir
        self.video_dir = base_dir.resolve()
        
        # Initialize directories using the same paths as VideoProcessor
        self.transcript_dir = (self.video_dir / "transcripts").resolve()
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summaries directory if it doesn't exist
        self.summaries_dir = (self.video_dir / "summaries").resolve()
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ContentAnalyzer with transcript_dir: {self.transcript_dir}")
        logger.info(f"Initialized ContentAnalyzer with summaries_dir: {self.summaries_dir}")
        
        # Get langtrace instance
        self.langtrace = get_langtrace()
        if self.langtrace:
            logger.info("LangTrace is available for tracing")
        else:
            logger.warning("LangTrace is not available, tracing will be disabled")

    async def _get_video_analysis_model(self):
        """Get or initialize video analysis model"""
        if not self.video_analysis_model:
            self.video_analysis_model = await self.model_manager.get_video_analysis_model()
        return self.video_analysis_model

    async def _analyze_transcript_segment(self, segment_text: str, parent_trace=None) -> Dict[str, Any]:
        """Analyze a transcript segment to determine its category and content"""
        try:
            # Wait for rate limit before proceeding
            model_name = self.settings.model.transcription.name
            await self.model_manager._wait_for_rate_limit(model_name)
            
            # Get the model for analysis
            model = await self.model_manager._initialize_model(model_name)
            
            if not segment_text or len(segment_text.strip()) < 10:
                logger.warning("Empty or very short segment, skipping analysis")
                return {
                    'category': 'general discussions',
                    'stock': None,
                    'summary': segment_text[:200] + "...",
                    'include': False
                }
                
            prompt = f"""You are a financial content analyzer. Analyze this video transcript segment and provide a detailed analysis.
            
            Instructions:
            1. Categorize the content into: 'stock analysis', 'daily news', or 'general discussions'
            2. For stock analysis, identify the specific stock ticker/name
            3. Provide a concise but informative summary of the key points
            4. Include any relevant market data or metrics mentioned
            5. Note any significant trends or patterns discussed
            
            Transcript segment:
            {segment_text}
            
            Format your response as follows:
            category: [category]
            stock: [stock ticker/name or 'none']
            summary: [your analysis]
            include: [true/false - indicate if this segment contains meaningful financial content]
            """
            
            # Call the model with tracing
            @trace_llm_call("analyze_segment")
            def analyze():
                # Generate content is synchronous, no need for await
                return model.generate_content(prompt)
                
            response = analyze()
            
            if not response or not response.text:
                logger.error(f"Empty response from model for segment")
                return {
                    'category': 'general discussions',
                    'stock': None,
                    'summary': f"Error analyzing segment: Empty response from model",
                    'include': False
                }
            
            # Parse and validate the response
            try:
                lines = response.text.strip().split('\n')
                result = {
                    'category': 'general discussions',  # default
                    'stock': None,
                    'summary': '',
                    'include': False
                }
                
                for line in lines:
                    line = line.strip().lower()
                    if line.startswith('category:'):
                        result['category'] = line.replace('category:', '').strip()
                    elif line.startswith('stock:'):
                        stock = line.replace('stock:', '').strip()
                        result['stock'] = None if stock in ['none', 'n/a', ''] else stock
                    elif line.startswith('summary:'):
                        result['summary'] = line.replace('summary:', '').strip()
                    elif line.startswith('include:'):
                        result['include'] = 'true' in line.lower()
                
                if self.langtrace is not None:
                    result['trace_id'] = parent_trace.trace_id if parent_trace else None
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to parse model response: {str(e)}")
                return {
                    'category': 'general discussions',
                    'stock': None,
                    'summary': f"Error analyzing segment: Failed to parse response: {str(e)}",
                    'include': False
                }
                
        except Exception as e:
            logger.error(f"Error analyzing segment: {str(e)}")
            return {
                'category': 'general discussions',
                'stock': None,
                'summary': f"Error analyzing segment: {str(e)}",
                'include': False
            }

    async def _analyze_frames(self, frames: List[str], category: str) -> Dict[str, Any]:
        """Analyze frames to find the best stock chart representations"""
        if category != 'stock analysis':
            return {'frames': [], 'trace_id': None}

        best_frames = []
        frame_batches = [frames[i:i+4] for i in range(0, len(frames), 4)]  # Process 4 frames at a time
        trace_id = None
        
        for batch in frame_batches:
            # Wait for rate limit before making request
            model_name = self.settings.model.video_analysis.name
            await self.model_manager._wait_for_rate_limit(model_name)
            
            # Load images
            images = []
            for frame_path in batch:
                try:
                    img = Image.open(frame_path)
                    images.append(img)
                except Exception as e:
                    logger.error(f"Failed to load image {frame_path}: {str(e)}")
                    continue

            if not images:
                continue

            prompt = """Analyze these stock chart images and rank them based on:
            1. Presence of technical analysis indicators
            2. Clarity of trend lines and patterns
            3. Timeframe visibility (daily, weekly, etc.)
            4. Overall information density
            
            Select the best 1-2 frames that show different timeframes or complementary analysis.
            Respond with ONLY the indices (0-based) of the best frames, separated by commas.
            Example: "0,2" means the 1st and 3rd frames are best.
            """
            
            try:
                model = await self.model_manager._initialize_model(model_name)
                
                # Analyze frames with tracing
                if self.langtrace is not None:
                    @trace_llm_call("analyze_frames")
                    async def analyze_frames():
                        return model.generate_content([prompt, *images])
                    response = await analyze_frames()
                else:
                    response = model.generate_content([prompt, *images])
                
                frame_indices = response.text.strip().split(',')
                
                # Add selected frames to best_frames
                for idx in frame_indices:
                    try:
                        idx = int(idx)
                        if 0 <= idx < len(batch):
                            best_frames.append(batch[idx])
                            if len(best_frames) >= 2:  # Don't select more than 2 frames
                                return {'frames': best_frames, 'trace_id': trace_id}
                    except ValueError:
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to analyze frames: {str(e)}")
                continue

        return {'frames': best_frames[:2], 'trace_id': trace_id}  # Return at most 2 frames

    async def analyze_video_content(self, video: Video) -> List[Dict[str, Any]]:
        """Analyze video content and generate structured summaries"""
        try:
            # Load frame mapping which contains both transcript and frame information
            mapping_path = self.transcript_dir / f"{video.id}_frame_mapping.json"
            
            if not mapping_path.exists():
                raise FileNotFoundError(f"Frame mapping file not found: {mapping_path}")
            
            logger.info(f"Loading frame mapping from {mapping_path}")
            with open(mapping_path) as f:
                frame_mapping = json.load(f)
            
            logger.info(f"Loaded frame mapping with {len(frame_mapping.get('mapping', []))} mapped segments")

            # Use LangTrace for overall analysis if enabled
            if self.langtrace is not None:
                @trace_llm_call("analyze_video_content")
                async def analyze_with_trace():
                    return await self._process_segments(video, frame_mapping)
                summaries = await analyze_with_trace()
            else:
                summaries = await self._process_segments(video, frame_mapping)

            logger.info(f"Generated {len(summaries)} content summaries")
            return summaries

        except Exception as e:
            logger.error(f"Failed to analyze video content: {str(e)}", exc_info=True)
            raise

    async def _process_segments(self, video: Video, frame_mapping: Dict, parent_trace=None) -> List[Dict[str, Any]]:
        """Process video segments and generate summaries"""
        summaries = []
        current_block = {
            "category": None,
            "stock": None,
            "text": "",
            "frames": [],
            "start_time": None,
            "end_time": None,
            "trace_ids": []
        }

        for segment in frame_mapping.get('mapping', []):
            if not segment.get('transcript'):
                continue

            # Analyze the segment with LangTrace context
            if self.langtrace is not None:
                @trace_llm_call("analyze_segment")
                async def analyze_segment():
                    return await self._analyze_transcript_segment(segment['transcript'], parent_trace)
                analysis = await analyze_segment()
            else:
                analysis = await self._analyze_transcript_segment(segment['transcript'])

            if analysis['include']:
                if current_block["category"] is None:
                    current_block.update({
                        "category": analysis["category"],
                        "stock": analysis["stock"],
                        "text": segment["transcript"],
                        "frames": segment.get("frames", []),
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"]
                    })
                elif current_block["category"] == analysis["category"] and current_block["stock"] == analysis["stock"]:
                    current_block["text"] += "\n" + segment["transcript"]
                    current_block["frames"].extend(segment.get("frames", []))
                    current_block["end_time"] = segment["end_time"]
                else:
                    # Process current block and start new one
                    if current_block["text"]:
                        summary = await self._process_block(video, current_block, parent_trace)
                        if summary:
                            summaries.append(summary)

                    current_block = {
                        "category": analysis["category"],
                        "stock": analysis["stock"],
                        "text": segment["transcript"],
                        "frames": segment.get("frames", []),
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "trace_ids": [analysis.get('trace_id')] if self.langtrace is not None else []
                    }

        # Process final block
        if current_block["text"]:
            summary = await self._process_block(video, current_block, parent_trace)
            if summary:
                summaries.append(summary)

        return summaries

    async def _process_block(self, video: Video, block: Dict, parent_trace=None) -> Optional[Dict[str, Any]]:
        """Process a content block and generate summary"""
        try:
            # Use video analysis model for generating summary
            model_name = self.settings.model.video_analysis.name
            await self.model_manager._wait_for_rate_limit(model_name)
            
            # Analyze frames if present
            frame_analysis = await self._analyze_frames(block["frames"], block["category"])
            
            # Generate summary prompt
            prompt = f"""Generate a detailed summary for this video segment about {block['category']}.
            
            Content: {block['text']}
            
            Format the response as:
            category: {block['category']}
            stock: {block['stock'] if block['stock'] else 'none'}
            start_time: {block['start_time']}
            end_time: {block['end_time']}
            key_points: [list 3-5 main points]
            summary: [2-3 sentences summarizing the content]
            """
            
            # Get langtrace instance
            langtrace_client = self.langtrace
            
            # Generate summary with tracing
            if langtrace_client is not None:
                @trace_llm_call("generate_summary")
                async def generate_summary():
                    model = await self.model_manager._initialize_model(model_name)
                    return model.generate_content(prompt)
                response = await generate_summary()
            else:
                model = await self.model_manager._initialize_model(model_name)
                response = model.generate_content(prompt)
            
            if not response or not response.text:
                logger.error("Empty response from model for summary generation")
                return None
            
            # Parse the response into a structured format
            summary = self._parse_summary_response(response.text)
            if not summary:
                logger.error("Failed to parse summary response")
                return None
            
            # Add metadata
            summary.update({
                "frames": frame_analysis.get("frames", []),
                "trace_ids": block.get("trace_ids", [])
            })
            
            # Save summary to file
            start_time_str = str(block['start_time']).replace(":", "-")  # Make filename safe
            summary_file = self.summaries_dir / f"{video.id}_{start_time_str}.json"
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved summary to {summary_file}")
            except Exception as e:
                logger.error(f"Failed to save summary file: {str(e)}")
                # Continue even if save fails - we still want to return the summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Error processing block: {str(e)}", exc_info=True)  # Add stack trace
            return None

    def _parse_summary_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the model's response into a structured format"""
        try:
            lines = response_text.strip().split('\n')
            summary = {}
            current_key = None
            current_value = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    # If we have a previous key, add it to the summary
                    if current_key and current_value:
                        summary[current_key] = '\n'.join(current_value).strip()
                        current_value = []
                    
                    # Start new key
                    key, value = line.split(':', 1)
                    current_key = key.strip().lower()
                    value = value.strip()
                    if value:
                        current_value.append(value)
                else:
                    # Continue previous value
                    if current_key:
                        current_value.append(line)
            
            # Add the last key-value pair
            if current_key and current_value:
                summary[current_key] = '\n'.join(current_value).strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error parsing summary response: {str(e)}")
            return None
