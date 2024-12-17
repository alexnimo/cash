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
        
        # Create raw transcript directory
        self.raw_transcript_dir = (self.video_dir / "raw_transcripts").resolve()
        self.raw_transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summaries directory if it doesn't exist
        self.summaries_dir = (self.video_dir / "summaries").resolve()
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ContentAnalyzer with transcript_dir: {self.transcript_dir}")
        logger.info(f"Initialized ContentAnalyzer with raw_transcript_dir: {self.raw_transcript_dir}")
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

    async def _upload_frame(self, frame_path: str) -> Optional[Any]:
        """Upload a single frame to Gemini using file API"""
        try:
            file = genai.upload_file(frame_path, mime_type="image/jpeg")
            logger.info(f"Uploaded frame '{file.display_name}' as: {file.uri}")
            return file
        except Exception as e:
            logger.error(f"Failed to upload frame {frame_path}: {str(e)}")
            return None

    async def _analyze_frames(self, frames: List[str], analysis: Dict) -> Dict[str, Any]:
        """Analyze frames to find relevant visual content for the analysis"""
        if not frames:
            return {'frames': []}

        try:
            model_name = self.settings.model.video_analysis.name
            await self.model_manager._wait_for_rate_limit(model_name)
            
            # Initialize model with vision capabilities
            model = await self.model_manager._initialize_model(model_name)
            
            # Upload all frames first
            logger.info(f"Starting upload of {len(frames)} frames")
            uploaded_frames = []
            frame_upload_tasks = []
            
            for frame_path in frames:
                frame_upload_tasks.append(self._upload_frame(frame_path))
            
            # Wait for all uploads to complete
            upload_results = await asyncio.gather(*frame_upload_tasks)
            
            # Filter out failed uploads
            uploaded_frames = [f for f in upload_results if f is not None]
            
            if not uploaded_frames:
                logger.error("No frames were successfully uploaded")
                return {'frames': []}
            
            logger.info(f"Successfully uploaded {len(uploaded_frames)} frames")
            
            # Process frames in batches
            max_batch_size = 5
            frame_batches = [uploaded_frames[i:i + max_batch_size] 
                           for i in range(0, len(uploaded_frames), max_batch_size)]
            
            all_frame_mappings = []
            
            # Create analysis context
            analysis_context = f"""
            Topic: {analysis.get('topic', '')}
            Stocks: {', '.join(analysis.get('stocks', []))}
            Summary: {analysis.get('summary', '')}
            Key Points: {', '.join(analysis.get('key_points', []))}
            """

            for batch_idx, frame_batch in enumerate(frame_batches):
                frame_indices = range(batch_idx * max_batch_size, 
                                   min((batch_idx + 1) * max_batch_size, len(frames)))
                
                prompt = f"""Analyze these frames in the context of the following financial analysis:
                {analysis_context}

                For each frame, determine if it:
                1. Shows relevant stock charts or technical patterns
                2. Displays price levels or movements discussed
                3. Illustrates market conditions or indicators
                4. Provides visual evidence supporting the analysis

                Return a JSON array of frame mappings, where each mapping includes:
                {{
                    'name': 'files/name',
                    'display_name': 'frame_number.jpg',
                    "content_type": "chart/indicator/price_level/other",
                    "description": "Brief description of what the frame shows",
                    "timestamp": "approximate timestamp in video",
                    "matching_points": ["Which key points this frame supports"]
                }}

                Only include frames that are relevant to the analysis.
                """

                # Create content list with prompt and frames
                content = [prompt]
                content.extend(frame_batch)

                if self.langtrace is not None:
                    @trace_llm_call("analyze_frames_batch")
                    def analyze_frames():
                        return model.generate_content(content)
                    response = analyze_frames()
                else:
                    response = model.generate_content(content)

                if not response or not response.text:
                    continue

                try:
                    batch_mappings = json.loads(response.text.strip())
                    if not isinstance(batch_mappings, list):
                        batch_mappings = [batch_mappings]
                    
                    # Add actual frame paths
                    for mapping, frame_idx in zip(batch_mappings, frame_indices):
                        mapping['frame_path'] = frames[frame_idx]
                        all_frame_mappings.append(mapping)

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse frame mapping from batch {batch_idx + 1}")
                    continue

            return {'frames': all_frame_mappings}

        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            return {'frames': []}

    async def _process_transcript(self, video: Video, raw_transcript: str) -> Dict[str, Any]:
        """Process raw transcript to identify stock discussions and their sections"""
        try:
            model_name = self.settings.model.video_analysis.name
            await self.model_manager._wait_for_rate_limit(model_name)
            model = await self.model_manager._initialize_model(model_name)
            
            prompt = f"""
            <instructions>
                <instruction>
            You are an expert financial content analyzer. 
            Analyze this video transcript about stock market analysis and trading.
            Break down the content into sections. Each section must be organized into one of the following topics:
            <stock analysis>, <general discussion>, <market news>, <market context>, <trading context>.
            For each topic, provide a concise but informative summary of the key points.
            When the topic is <stock analysis>, include the specific stock ticker/name and any relevant market data or metrics mentioned.
            Note any significant trends or patterns discussed.
            Break down the content by stock tickers discussed and provide detailed analysis for each.
                </instruction>
                <instruction>
               - Provide a dedicated section for each stock/ticker discussed.
               - Provide a dedicated section for each section.
                </instruction>
                <instruction>
                When you have identified a non relevant discussion, comercial, advertising, etc, do not include it in your analysis.
                </instruction>

            </instructions>


            Transcript:
            {raw_transcript}

            For each stock or group of stocks discussed, provide analysis in this format:
            {{
                "topic": "",      // Category of discussion
                "stocks": ["TICKER"],  // Stock ticker discussed
                "start_time": "timestamp",         // Approximate start time of discussion
                "end_time": "timestamp",           // Approximate end time of discussion
                "sections": 
                "summary": "",
                "key_points": [
                        "Specific price levels",
                        "Technical patterns",
                        "Trading signals",
                        "Timeframes"
                        ],
                "overall_summary": "Main trading thesis and key takeaways"
            }}
            """

            if self.langtrace is not None:
                @trace_llm_call("process_transcript")
                def analyze():
                    return model.generate_content(prompt)
                response = analyze()
            else:
                response = model.generate_content(prompt)

            if not response or not response.text:
                logger.error("Empty response from model")
                return {"raw_response": "", "stocks": []}

            # Save raw response
            raw_response_path = self.summaries_dir / f"{video.id}_raw_llm.txt"
            with open(raw_response_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Saved raw LLM response to {raw_response_path}")

            try:
                # Try to parse as JSON but don't modify the structure
                parsed_response = json.loads(response.text.strip())
                return {"raw_response": response.text, "stocks": parsed_response if isinstance(parsed_response, list) else [parsed_response]}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse as JSON: {str(e)}")
                return {"raw_response": response.text, "stocks": []}

        except Exception as e:
            logger.error(f"Failed to process transcript: {str(e)}", exc_info=True)
            return {"raw_response": "", "stocks": []}

    async def analyze_video_content(self, video: Video) -> Dict[str, Any]:
        """Analyze video content and generate structured summaries"""
        try:
            # Load raw transcript
            raw_transcript_path = self.raw_transcript_dir / f"{video.id}.txt"
            if not raw_transcript_path.exists():
                raise FileNotFoundError(f"Raw transcript file not found: {raw_transcript_path}")
            
            logger.info(f"Loading raw transcript from {raw_transcript_path}")
            with open(raw_transcript_path) as f:
                raw_transcript = f.read()

            # Process transcript and get raw response
            analysis = await self._process_transcript(video, raw_transcript)
            
            # Save raw LLM output with metadata
            raw_analysis = {
                "video_id": video.id,
                "title": video.metadata.title if video.metadata.title else video.id,
                "duration": video.metadata.duration,
                "analysis_timestamp": datetime.now().isoformat(),
                "raw_llm_response": analysis["raw_response"]
            }
            
            raw_output_path = self.summaries_dir / f"{video.id}_raw.json"
            with open(raw_output_path, 'w') as f:
                json.dump(raw_analysis, f, indent=2)
            logger.info(f"Saved raw analysis to {raw_output_path}")

            # Only proceed with frame mapping if we have valid JSON
            if analysis["stocks"]:
                # Load frames information
                frames_path = self.video_dir / "frames" / str(video.id)
                if frames_path.exists():
                    frame_files = [str(f) for f in frames_path.glob("*.jpg")]
                    frame_files.sort()
                    
                    # Add frame analysis
                    for stock_analysis in analysis["stocks"]:
                        if frame_files:
                            frame_analysis = await self._analyze_frames(frame_files, stock_analysis)
                            stock_analysis['frames'] = frame_analysis['frames']

                    # Save combined analysis with frames
                    output_path = await self._save_combined_analysis(video, {"stocks": analysis["stocks"]})
                    logger.info(f"Saved combined analysis to {output_path}")
                    
                    return {
                        "raw_output_path": str(raw_output_path),
                        "analysis_path": output_path,
                        "stock_count": len(analysis["stocks"]),
                        "stocks": [stock.get('stocks', []) for stock in analysis["stocks"]]
                    }

            return {
                "raw_output_path": str(raw_output_path),
                "stock_count": 0,
                "stocks": []
            }

        except Exception as e:
            logger.error(f"Failed to analyze video content: {str(e)}", exc_info=True)
            raise

    async def _save_combined_analysis(self, video: Video, analysis: Dict[str, Any]) -> str:
        """Save all analysis data in a single JSON file"""
        try:
            combined_analysis = {
                "video_id": video.id,
                "title": video.metadata.title if video.metadata.title else video.id,
                "duration": video.metadata.duration,
                "analysis_timestamp": datetime.now().isoformat(),
                "stocks": []
            }

            # Restructure each stock analysis
            for stock_data in analysis['stocks']:
                stock_entry = {
                    "topic": stock_data.get('topic', ''),
                    "stocks": stock_data.get('stocks', []),
                    "start_time": stock_data.get('start_time', '0'),
                    "end_time": stock_data.get('end_time', str(video.metadata.duration)),
                    "summary": stock_data.get('summary', ''),
                    "key_points": stock_data.get('key_points', []),
                    "overall_summary": stock_data.get('overall_summary', ''),
                    "frames": stock_data.get('frames', [])
                }
                combined_analysis["stocks"].append(stock_entry)

            # Save to file
            output_path = self.summaries_dir / f"{video.id}.json"
            with open(output_path, 'w') as f:
                json.dump(combined_analysis, f, indent=2)

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save combined analysis: {str(e)}")
            raise

    def _create_default_analysis(self, block: Dict) -> Dict[str, Any]:
        """Create a default analysis when the model fails to generate a valid one"""
        return {
            "topic": block.get("topic", "general discussion"),
            "stocks": block.get("stocks", ["unknown"]),
            "summary": "Default analysis of market conditions",
            "key_points": [
                "Market conditions",
                "Trading considerations",
                "Technical patterns",
                "Price levels"
            ],
            "overall_summary": "Analysis of market conditions and trading opportunities",
            "frames": []
        }
