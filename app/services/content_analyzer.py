import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from datetime import datetime, timedelta
from app.core.config import get_settings
from app.models.video import Video, VideoStatus
from app.services.model_manager import ModelManager
from app.services.report_generator import ReportGenerator
from app.utils.langtrace_utils import get_langtrace, trace_llm_call, init_langtrace
from PIL import Image
import os
import base64
import PyPDF2
import math
import tempfile
import re
from app.services.errors import VideoProcessingError
from app.services.video_processor import Video

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
        
        # Initialize langtrace
        init_langtrace()
        self.langtrace = get_langtrace()
        if self.langtrace:
            logger.info("LangTrace is available for tracing")
            logger.info(f"Initialized with dirs: transcript_dir={self.transcript_dir}, raw_transcript_dir={self.raw_transcript_dir}, summaries_dir={self.summaries_dir}, reports_dir={self.reports_dir}")
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
            - Use proper JSON formatting as shown in the example
            
            """
            
            if self.langtrace is not None:
                @trace_llm_call("analyze_transcript")
                def analyze():
                    return model.generate_content(prompt)
                response = analyze()
            else:
                response = model.generate_content(prompt)
            
            if self.langtrace is not None:
                with self.langtrace.start_span("analyze_transcript_response") as span:
                    span.set_attribute("response_length", len(response.text))
                    span.set_attribute("response_preview", response.text[:200])
            
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

    async def analyze_frames_batch(self, frames: List[str], retry_count: int = 0) -> Dict[str, Any]:
        """Upload and analyze a batch of frames with rate limit handling"""
        try:
            # Upload frames
            uploaded_frames = []
            for frame in frames:
                try:
                    uploaded_frame = await self._upload_frame(frame)
                    uploaded_frames.append(uploaded_frame)
                except Exception as e:
                    logger.error(f"Failed to upload frame {frame}: {str(e)}")
                    if retry_count < 3:
                        logger.info(f"Retrying frame upload, attempt {retry_count + 1}")
                        return await self.analyze_frames_batch(frames, retry_count + 1)
                    else:
                        logger.error("Max retries reached for frame upload")
                        return {"error": str(e)}
            
            return {"frames": uploaded_frames}
            
        except Exception as e:
            logger.error(f"Failed to analyze frames batch: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def _split_pdf_and_upload(self, pdf_path: str) -> List[str]:
        """Split a PDF into chunks and upload each chunk."""
        try:
            # Read the PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                # Calculate chunk size (max 10 pages per chunk)
                chunk_size = 10
                num_chunks = math.ceil(num_pages / chunk_size)
                
                # Create temporary directory for chunks
                with tempfile.TemporaryDirectory() as temp_dir:
                    chunk_files = []
                    
                    # Split PDF into chunks
                    for i in range(num_chunks):
                        start_page = i * chunk_size
                        end_page = min((i + 1) * chunk_size, num_pages)
                        
                        # Create PDF writer for this chunk
                        pdf_writer = PyPDF2.PdfWriter()
                        
                        # Add pages to chunk
                        for page_num in range(start_page, end_page):
                            pdf_writer.add_page(pdf_reader.pages[page_num])
                        
                        # Save chunk to temporary file
                        chunk_path = os.path.join(temp_dir, f'chunk_{i}.pdf')
                        with open(chunk_path, 'wb') as chunk_file:
                            pdf_writer.write(chunk_file)
                        
                        chunk_files.append(chunk_path)
                        logger.info(f"Created chunk {i+1}/{num_chunks} at {chunk_path}")
                    
                    # Upload each chunk
                    uploaded_files = []
                    for chunk_path in chunk_files:
                        try:
                            file = genai.upload_file(chunk_path, mime_type="application/pdf")
                            uploaded_files.append(file)
                            logger.info(f"Uploaded chunk {chunk_path}")
                        except Exception as e:
                            logger.error(f"Failed to upload chunk {chunk_path}: {str(e)}")
                            continue
                    
                    return uploaded_files
                    
        except Exception as e:
            logger.error(f"Failed to split and upload PDF: {str(e)}")
            return []

    @trace_llm_call("analyze_frames_and_update_summary")
    async def _analyze_frames_and_update_summary(self, video: Video, frames_pdf_path: str, summary_json_path: str) -> Dict[str, Any]:
        """Analyze frames from PDF and update summary with best frames for each section."""
        try:
            # Get the model
            model = await self._get_video_analysis_model()
            
            # Load the summary
            with open(summary_json_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            try:
                # Split and upload PDF files
                pdf_files = await self._split_pdf_and_upload(frames_pdf_path)
                logger.info(f"Successfully uploaded {len(pdf_files)} PDF chunks")
                
                # Create parts with all PDF files
                prompt = f"""
                You are an expert financial content analyzer. 
                Your task is to analyze frames from a financial video that have been split into {len(pdf_files)} PDF chunks.
                Each chunk contains a portion of the frames from the video.
                
                Section = {{
                    'topic': str,  # Must be one of: ["stock analysis", "general discussion", "market news", "market context", "trading context"]
                    'stocks': list[str],  # Empty list if no stocks mentioned: []
                    'start_time': float,  # Time in seconds, e.g., 0.16
                    'end_time': float,  # Time in seconds, e.g., 40.92
                    'summary': str,  # Brief summary of the section
                    'key_points': list[str],  # List of key points as strings
                    'overall_summary': str,  # Overall section summary
                    'frame_paths': list[str]  # List of full frame paths from PDF metadata
                }}
                
                Return: list[Section]
                
                Instructions for frame analysis:
                1. For each frame in ALL PDF chunks:
                   - Look for stock chart graphs and analyze:
                     * Stock tickers visible in the chart
                     * Timeframe of the chart (daily, weekly, etc.)
                     * Technical analysis elements (trendlines, support/resistance, etc.)
                     * Visual quality and readability
                   - Record the full path of the frame from the PDF metadata
                   - Assign a quality score (0-10) based on clarity and information
                   - Pick only one best frame for each section based on the sections above
            
                2. For each section in the summary:
                   - If the section discusses specific stocks:
                     * Find frames showing those exact stocks
                     * For each stock, select the highest quality frame
                     * Add the FULL FRAME PATHS to the section's frame_paths list
                   - If no specific stocks discussed, leave frame_paths as empty list
            
                3. CRITICAL: You MUST:
                   - Use DOUBLE QUOTES for all strings and property names
                   - Use the exact frame paths as shown in the PDF metadata
                   - DO NOT modify, shorten, or create new paths
                   - Return valid JSON matching the schema exactly
                   - Include all fields for each section
                   - Use proper JSON formatting as shown in the example
            
                Example of valid response:
                [
                    {{
                        "topic": "general discussion",
                        "stocks": [],
                        "start_time": 0.16,
                        "end_time": 40.92,
                        "summary": "Discussion about market trends",
                        "key_points": ["Point 1", "Point 2"],
                        "overall_summary": "Overview of market conditions",
                        "frame_paths": []
                    }},
                    {{
                        "topic": "stock analysis",
                        "stocks": ["AAPL", "MSFT"],
                        "start_time": 41.0,
                        "end_time": 120.5,
                        "summary": "Analysis of tech stocks",
                        "key_points": ["Tech sector growth", "Earnings forecast"],
                        "overall_summary": "Positive outlook for tech sector",
                        "frame_paths": [
                            "C:\\Users\\A&O\\frames\\frame_123.jpg",
                            "C:\\Users\\A&O\\frames\\frame_124.jpg"
                        ]
                    }}
                ]
                
                Here is the existing summary to update with frames:
                {json.dumps(summary, indent=2)}
                
                Analyze ALL frames from ALL PDF chunks and return the complete updated summary following the schema above.
                Store your response in a variable called 'result' and return that variable.
                """
                
                # Wait for rate limit before proceeding
                model_name = self.settings.model.transcription.name
                await self.model_manager._wait_for_rate_limit(model_name)
                
                try:
                    logger.info("Starting frame analysis with all PDF chunks...")
                    # Generate content with all parts
                    response = await asyncio.to_thread(
                        model.generate_content,
                        contents=[{"text": prompt}] + [{"file_data": pdf_file} for pdf_file in pdf_files],
                        generation_config={
                            'temperature': 0.1,
                            'top_p': 0.1,
                            'top_k': 40
                        }
                    )
                    logger.info("Frame analysis completed successfully")
                    
                    # Log raw response for debugging
                    response_text = response.text.strip()
                    
                    # Save raw response
                    raw_response_path = self.summaries_dir / f"{video.id}_raw_frames_analysis.txt"
                    with open(raw_response_path, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    logger.info(f"Saved raw frames analysis to {raw_response_path}")
                    
                    try:
                        # Remove markdown code block markers
                        if "```json" in response_text:
                            response_text = response_text.replace("```json", "").replace("```", "").strip()
                        elif "```" in response_text:
                            lines = response_text.split("\n")
                            lines = [line for line in lines if not line.startswith("```") and not line.endswith("```")]
                            response_text = "\n".join(lines).strip()
                        
                        # Parse JSON response
                        updated_sections = json.loads(response_text)
                        
                        # Validate sections is a list
                        if not isinstance(updated_sections, list):
                            logger.error("Response is not a list of sections")
                            summary["sections"] = response_text  # Save raw response
                            return summary
                        
                        # Update the summary with frame information
                        summary["sections"] = updated_sections
                        return summary
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse frame analysis JSON: {str(e)}")
                        # Save the raw response in the summary instead of failing
                        summary["sections"] = response_text
                        return summary
                    
                except Exception as e:
                    logger.error(f"Failed to analyze frames and update summary: {str(e)}", exc_info=True)
                    return summary
                
            except Exception as e:
                logger.error(f"Failed to analyze frames and update summary: {str(e)}")
                return summary
            
        except Exception as e:
            logger.error(f"Failed to analyze frames and update summary: {str(e)}", exc_info=True)
            return summary

    def _clean_json_response(self, response_text: str) -> str:
        """Clean and prepare JSON response for parsing."""
        try:
            # Find content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            
            start_idx = response_text.find(start_marker)
            if start_idx != -1:
                # Move past the start marker
                start_idx += len(start_marker)
                end_idx = response_text.find(end_marker, start_idx)
                if end_idx != -1:
                    return response_text[start_idx:end_idx].strip()
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {str(e)}")
            logger.debug(f"Problematic response text: {response_text[:200]}...")
            return response_text

    async def _process_transcript(self, video: Video, raw_transcript: str) -> Dict[str, Any]:
        """Process the transcript and generate a summary."""
        try:
            # Get the model
            model = await self._get_video_analysis_model()
            
            # Define the schema in the prompt
            prompt = f"""
            You are an expert financial content analyzer. 
            Your task is to analyze this video transcript about stock market analysis and trading.
            
            Use this JSON schema:
            Section = {{
                'topic': str,  # One of: "stock analysis", "general discussion", "market news", "market context", "trading context"
                'stocks': list[str],
                'start_time': float,
                'end_time': float,
                'summary': str,
                'key_points': list[str],
                'overall_summary': str
            }}
            
            Return: list[Section]
            
            Instructions:
            1. Break down the content into sections based on the topics listed above
            2. For each section provide all fields defined in the schema
            3. When analyzing stock-specific sections:
               - Include specific stock tickers/names
               - Note any market data or metrics mentioned
               - Highlight significant trends or patterns
            4. Important:
               - Skip any non-relevant discussions, commercials, or advertising
               - Focus on actionable trading insights and market analysis
               - Be specific and detailed in your summaries
               - Use proper stock ticker symbols
            
            IMPORTANT: Your response must be a valid JSON object matching the schema above.
            Do not include any text before or after the JSON.
            
            Here is the transcript to analyze:
            {raw_transcript}
            """

            # Generate content
            response = await asyncio.to_thread(
                model.generate_content,
                contents=[{"text": prompt}],
                generation_config={
                    'temperature': 0.1,
                    'top_p': 0.1,
                    'top_k': 40
                }
            )
            
            # Log raw response for debugging
            response_text = response.text.strip()
            logger.info(f"Raw response from model: {response_text[:500]}...")
            
            # Save raw response
            raw_response_path = self.summaries_dir / f"{video.id}_raw_response.txt"
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.info(f"Saved raw response to {raw_response_path}")
            
            try:
                # Clean and parse JSON
                response_text = self._clean_json_response(response_text)
                sections = json.loads(response_text)
                
                # Validate sections is a list
                if not isinstance(sections, list):
                    logger.error("Response is not a list of sections")
                    sections = [{"topic": "Parsing Error", "stocks": [], "start_time": 0, "end_time": 0, 
                               "summary": response_text, "key_points": [], "overall_summary": "Failed to parse response"}]
                
                # Create the analysis object
                analysis_json = {
                    "Date": datetime.now().strftime("%B %d %Y"),
                    "Channel name": video.metadata.channel_name if video.metadata else "",
                    "Video name": video.metadata.video_title if video.metadata else "",
                    "sections": sections
                }
                
                # Create overall summary from section summaries
                overall_summary = "\n".join([
                    section.get("overall_summary", "")
                    for section in sections
                ])
                
                # Return the full analysis
                return {
                    "raw_response": overall_summary,
                    "stocks": sections,  # Keep original sections for compatibility
                    "analysis_json": analysis_json
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON: {str(e)}")
                sections = [{"topic": "Parsing Error", "stocks": [], "start_time": 0, "end_time": 0, 
                           "summary": response_text, "key_points": [], "overall_summary": f"Failed to parse: {str(e)}"}]
                
                analysis_json = {
                    "Date": datetime.now().strftime("%B %d %Y"),
                    "Channel name": video.metadata.channel_name if video.metadata else "",
                    "Video name": video.metadata.video_title if video.metadata else "",
                    "sections": sections
                }
                
                # Save the error response for debugging
                error_response_path = self.summaries_dir / f"{video.id}_error_response.txt"
                with open(error_response_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error parsing JSON: {str(e)}\n\nRaw response:\n{response_text}")
                logger.error(f"Saved error response to {error_response_path}")
                
                # Return valid structure with raw response
                return {
                    "raw_response": f"Failed to parse: {str(e)}",
                    "stocks": sections,
                    "analysis_json": analysis_json
                }
            
        except Exception as e:
            logger.error(f"Error in transcript processing: {str(e)}")
            raise VideoProcessingError("Failed to analyze transcript content")

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
            
            # Save the initial analysis JSON
            if 'analysis_json' in analysis:
                initial_output_path = await self._save_combined_analysis(video, analysis['analysis_json'])
                logger.info(f"Saved initial analysis to {initial_output_path}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze video content: {str(e)}")
            raise

    async def _save_combined_analysis(self, video: Video, analysis: Dict[str, Any]) -> str:
        """Save all analysis data in a single JSON file."""
        try:
            # Save the analysis JSON
            output_path = self.summaries_dir / f"{video.id}_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved analysis to {output_path}")
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
