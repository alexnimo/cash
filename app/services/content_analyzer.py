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
from app.services.report_generator import ReportGenerator
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

        # Create reports directory if it doesn't exist
        self.reports_dir = (self.video_dir / "reports").resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize report generator
        self.report_generator = ReportGenerator(str(self.reports_dir))
        
        logger.info(f"Initialized ContentAnalyzer with transcript_dir: {self.transcript_dir}")
        logger.info(f"Initialized ContentAnalyzer with raw_transcript_dir: {self.raw_transcript_dir}")
        logger.info(f"Initialized ContentAnalyzer with summaries_dir: {self.summaries_dir}")
        logger.info(f"Initialized ContentAnalyzer with reports_dir: {self.reports_dir}")
        
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

    async def _analyze_transcript_segment(self, transcript_text: str, parent_trace=None) -> List[Dict[str, Any]]:
        """Analyze the full transcript and break it into meaningful segments"""
        try:
            # Wait for rate limit before proceeding
            model_name = self.settings.model.transcription.name
            await self.model_manager._wait_for_rate_limit(model_name)
            
            # Get the model for analysis
            model = await self.model_manager._initialize_model(model_name)
            
            if not transcript_text or len(transcript_text.strip()) < 10:
                logger.warning("Empty or very short transcript, skipping analysis")
                return []
                
            prompt = f"""You are a financial content analyzer. Analyze this complete video transcript and break it down into meaningful segments.
            The transcript includes timestamps in [HH:MM:SS] format. Use these to determine segment boundaries.

            Instructions:
            1. Read and understand the entire transcript to get the full context
            2. Break it down into logical segments based on topic changes and content
            3. For each segment provide:
               - Topic category (e.g., 'market analysis', 'stock analysis', 'general discussion')
               - All stocks mentioned (use ticker symbols)
               - Precise timestamps from the transcript
               - Detailed summary of what was discussed
               - Key points and insights
               - Overall conclusion for the segment

            Transcript:
            {transcript_text}

            Format your response as a JSON array of segments. Each segment must have:
            - topic: "<category>"
            - stocks: [list of stock tickers]
            - start_time: "MM:SS" (from transcript timestamps)
            - end_time: "MM:SS" (from transcript timestamps)
            - summary: "detailed description of what was discussed"
            - key_points: ["point 1", "point 2", ...]
            - overall_summary: "conclusion about this segment"

            Example segment:
            {{
                "topic": "<market analysis>",
                "stocks": ["SPY", "QQQ"],
                "start_time": "0:07",
                "end_time": "0:47",
                "summary": "The S&P 500 closed down, showing limited price movement...",
                "key_points": [
                    "Key support at 4594",
                    "Potential double top bearish pattern forming",
                    "Break below support could lead to 4588"
                ],
                "overall_summary": "S&P 500 showing signs of weakness with potential bearish pattern"
            }}

            Important:
            - Use the actual timestamps from the transcript
            - Include all relevant stocks mentioned
            - Make segments based on natural topic transitions
            - Ensure all fields are present in each segment
            """
            
            # Call the model with tracing
            @trace_llm_call("analyze_transcript")
            def analyze():
                return model.generate_content(prompt)
                
            response = analyze()
            
            if not response or not response.text:
                logger.error(f"Empty response from model")
                return []
            
            # Parse and validate the response
            try:
                # Try to parse the response as JSON
                response_text = response.text.strip()
                # Find the first [ and last ] to extract the JSON array
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start == -1 or end == 0:
                    raise ValueError("No JSON array found in response")
                    
                json_text = response_text[start:end]
                segments = json.loads(json_text)
                
                if not isinstance(segments, list):
                    raise ValueError("Response is not a list of segments")
                    
                # Validate each segment has required fields
                for segment in segments:
                    required_fields = ['topic', 'stocks', 'start_time', 'end_time', 'summary', 'key_points', 'overall_summary']
                    missing_fields = [field for field in required_fields if field not in segment]
                    if missing_fields:
                        raise ValueError(f"Segment missing required fields: {missing_fields}")
                
                logger.info(f"Successfully parsed {len(segments)} segments")
                return segments
                
            except Exception as e:
                logger.error(f"Failed to parse model response: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            return []

    async def _upload_frame(self, frame_path: str) -> Optional[Any]:
        """Upload a single frame to Gemini using file API"""
        try:
            file = genai.upload_file(frame_path, mime_type="image/jpeg")
            logger.info(f"Uploaded frame '{file.display_name}' as: {file.uri}")
            return file
        except Exception as e:
            logger.error(f"Failed to upload frame {frame_path}: {str(e)}")
            return None

    async def analyze_frames_batch(self, frames: List[str], retry_count: int = 0) -> List[Dict]:
        """Upload and analyze a batch of frames with rate limit handling"""
        try:
            model = await self._get_video_analysis_model()
            results = []
            
            for frame in frames:
                try:
                    image = await self._upload_frame(frame)
                    
                    prompt = """Analyze this frame from a stock market video.
                    Extract and provide the following information in JSON format:
                    1. Stock tickers visible (if any)
                    2. Timeframe of the chart (if visible)
                    3. Technical analysis elements:
                       - Trendlines
                       - Support/Resistance levels
                       - Indicators visible
                       - Pattern formations
                    4. Graph visibility metrics:
                       - Clarity (0-10)
                       - Data completeness (0-10)
                       - Text readability (0-10)
                    5. Frame quality:
                       - Resolution
                       - Cropping/framing
                       - Any overlays or obstructions
                    
                    Also provide an overall rank (0-10) based on information completeness and visual quality.
                    """
                    
                    response = await model.generate_content([prompt, image])
                    analysis = json.loads(response.text)
                    
                    results.append({
                        'frame_path': frame,
                        'analysis': analysis,
                        'timestamp': float(frame.split('_')[1].split('.')[0])
                    })
                    
                except Exception as e:
                    if "429" in str(e) and retry_count < self.settings.rate_limit.max_retries:
                        logger.warning(f"Rate limit hit, waiting {self.settings.rate_limit.retry_delay_seconds} seconds...")
                        await asyncio.sleep(self.settings.rate_limit.retry_delay_seconds)
                        # Retry this batch from the failed frame
                        remaining_frames = frames[frames.index(frame):]
                        retry_results = await self.analyze_frames_batch(remaining_frames, retry_count + 1)
                        results.extend(retry_results)
                        break
                    else:
                        logger.error(f"Error analyzing frame {frame}: {str(e)}")
                        continue
            
            return results
        except Exception as e:
            logger.error(f"Failed to analyze frames batch: {str(e)}")
            return []

    async def analyze_frames_for_topics(self, video: Video, frames: List[str], transcript_analysis: Dict) -> Dict[str, List[Dict]]:
        """Analyze all frames and select the best ones for each topic/ticker"""
        try:
            logger.info(f"Starting comprehensive frame analysis for video {video.id}")
            
            # Process frames in batches
            all_frame_analyses = []
            batch_size = self.settings.rate_limit.batch_size
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                batch_results = await self.analyze_frames_batch(batch)
                all_frame_analyses.extend(batch_results)
            
            # Group frames by stock tickers
            ticker_frames = {}
            for frame_analysis in all_frame_analyses:
                tickers = frame_analysis['analysis'].get('stock_tickers', [])
                for ticker in tickers:
                    if ticker not in ticker_frames:
                        ticker_frames[ticker] = []
                    ticker_frames[ticker].append(frame_analysis)
            
            # For each topic in the transcript, select the best frames
            topic_best_frames = {}
            
            for segment in transcript_analysis['segments']:
                topic_key = f"{segment['topic']}_{','.join(segment['stocks'])}"
                relevant_frames = []
                
                # Get frames within segment timeframe
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Collect frames for all mentioned stocks in this segment
                for stock in segment['stocks']:
                    if stock in ticker_frames:
                        stock_frames = ticker_frames[stock]
                        # Filter frames by timestamp and sort by rank
                        segment_frames = [
                            frame for frame in stock_frames
                            if start_time <= frame['timestamp'] <= end_time
                        ]
                        segment_frames.sort(key=lambda x: x['analysis']['overall_rank'], reverse=True)
                        
                        # Get up to 2 best frames with different timeframes
                        selected_frames = []
                        seen_timeframes = set()
                        
                        for frame in segment_frames:
                            timeframe = frame['analysis'].get('timeframe')
                            if timeframe and timeframe not in seen_timeframes and len(selected_frames) < 2:
                                selected_frames.append(frame)
                                seen_timeframes.add(timeframe)
                        
                        relevant_frames.extend(selected_frames)
                
                if relevant_frames:
                    topic_best_frames[topic_key] = {
                        'frames': relevant_frames,
                        'topic': segment['topic'],
                        'stocks': segment['stocks'],
                        'timeframes': list(seen_timeframes),
                        'segment_time': {
                            'start': start_time,
                            'end': end_time
                        }
                    }
            
            return topic_best_frames
            
        except Exception as e:
            logger.error(f"Failed to analyze frames for topics: {str(e)}")
            return {}

    async def _process_transcript(self, video: Video, raw_transcript: str) -> Dict[str, Any]:
        """Process raw transcript to identify stock discussions and their sections"""
        try:
            model_name = self.settings.model.video_analysis.name
            await self.model_manager._wait_for_rate_limit(model_name)
            model = await self.model_manager._initialize_model(model_name)
            
            prompt = f"""
            instructions:
            You are an expert financial content analyzer. 
            Analyze this video transcript about stock market analysis and trading.
            Break down the content into sections. Each section must be organized into one of the following topics:
            stock analysis, general discussion, market news, market context, trading context.
            For each topic, provide a concise but informative summary of the key points.
            When the topic is "stock analysis", include the specific stock ticker/name and any relevant market data or metrics mentioned.
            Note any significant trends or patterns discussed.
            Break down the content by stock tickers discussed and provide detailed analysis for each.
            
               - Provide a dedicated section for each stock/ticker discussed.
               - Provide a dedicated section for each topic.
                When you have identified a non relevant discussion, comercial, advertising, general discussion, do not include it in your analysis.

            Transcript:
            {raw_transcript}

            Output the data in a valid json format with the following structure:
            {{
                "Date": "",
                "sections": [
                    {{
                        "topic": "",
                        "stocks": [],
                        "start_time": 0,
                        "end_time": 10,
                        "summary": "Summary of the stock analysis section",
                        "key_points": ["Point 1", "Point 2"],
                        "overall_summary": "Overall summary of the stock analysis section"
                    }}
                ],
            }}

            """

            if self.langtrace is not None:
                @trace_llm_call("process_transcript")
                def analyze():
                    return model.generate_content(prompt)
                response = analyze()
            else:
                response = model.generate_content(prompt)

            try:
                # Extract JSON content from response
                text = response.text.strip()
                # Find the first { and last } to extract JSON
                start = text.find('{')
                end = text.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = text[start:end]
                    # Attempt to parse JSON to validate it
                    parsed_response = json.loads(json_str)
                    
                    # Save the properly formatted JSON
                    raw_response_path = self.summaries_dir / f"{video.id}_raw_llm.json"
                    with open(raw_response_path, 'w') as f:
                        json.dump(parsed_response, f, indent=2)
                    logger.info(f"Saved formatted JSON response to {raw_response_path}")
                    
                    return {
                        "raw_response": parsed_response.get("overall_summary", ""),
                        "stocks": parsed_response.get("sections", []),
                        "analysis_json": parsed_response  # Include the full parsed JSON
                    }
                else:
                    logger.error("No JSON content found in response")
                    # Save the raw text for debugging
                    raw_response_path = self.summaries_dir / f"{video.id}_raw_llm.txt"
                    with open(raw_response_path, 'w') as f:
                        f.write(text)
                    logger.error(f"Saved problematic response to {raw_response_path}")
                    return {"raw_response": text, "stocks": [], "analysis_json": {}}
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse as JSON: {str(e)}")
                # Save the raw text for debugging
                raw_response_path = self.summaries_dir / f"{video.id}_raw_llm.txt"
                with open(raw_response_path, 'w') as f:
                    f.write(text)
                logger.error(f"Saved problematic response to {raw_response_path}")
                return {"raw_response": response.text, "stocks": [], "analysis_json": {}}
                
        except Exception as e:
            logger.error(f"Failed to process transcript: {str(e)}", exc_info=True)
            return {"raw_response": "", "stocks": [], "analysis_json": {}}

    async def analyze_video_content(self, video: Video) -> Dict[str, Any]:
        """Analyze video content and generate structured summaries"""
        try:
            logger.info(f"Starting content analysis for video {video.id}")
            
            # Build full transcript text with timestamps
            full_transcript = []
            for segment in video.transcript:
                timestamp = f"[{str(timedelta(seconds=int(segment.start_time)))}]"
                full_transcript.append(f"{timestamp} {segment.text}")
            
            full_transcript_text = " ".join(full_transcript)
            logger.info(f"Processing full transcript of length {len(full_transcript_text)}")
            
            # Process the transcript and get analysis
            analysis = await self._process_transcript(video, full_transcript_text)
            
            # Save the analysis JSON and generate frames report
            if 'analysis_json' in analysis:
                output_path = await self._save_combined_analysis(video, analysis['analysis_json'])
                logger.info(f"Saved combined analysis to {output_path}")
        
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze video content: {str(e)}")
            raise

    async def _save_combined_analysis(self, video: Video, analysis: Dict[str, Any]) -> str:
        """Save all analysis data in a single JSON file and generate frames report"""
        try:
            output_path = self.summaries_dir / f"{video.id}_analysis.json"
            
            # Add metadata
            final_analysis = {
                "metadata": {
                    "video_id": video.id,
                    "title": video.metadata.title if video.metadata.title else video.id,
                    "url": str(video.url),
                    "duration": video.metadata.duration if video.metadata else None,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            # Add the analyses
            final_analysis.update(analysis)
            
            with open(output_path, 'w') as f:
                json.dump(final_analysis, f, indent=2)
                
            logger.info(f"Saved analysis to {output_path}")

            # Generate frames report after saving analysis
            try:
                logger.info(f"Attempting to generate frames report for video {video.id}")
                logger.debug(f"Video analysis state: {video.analysis if hasattr(video, 'analysis') else 'No analysis'}")
                logger.debug(f"Video key frames: {video.analysis.key_frames if hasattr(video, 'analysis') and hasattr(video.analysis, 'key_frames') else 'No key_frames'}")
                
                report_path = self.report_generator.generate_frames_report(video)
                logger.info(f"Successfully generated frames report at {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate frames report: {str(e)}", exc_info=True)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
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
