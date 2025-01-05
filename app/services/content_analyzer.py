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
                    
                    if self.langtrace is not None:
                        @trace_llm_call("analyze_frame")
                        def analyze():
                            return model.generate_content(prompt, image)
                        response = analyze()
                    else:
                        response = model.generate_content(prompt, image)
                    
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

    async def _split_pdf_and_upload(self, pdf_path: str, max_size_mb: int = 35) -> List[Any]:
        """Split a PDF into chunks of max_size_mb and upload each chunk to Gemini."""
        max_size_bytes = max_size_mb * 1024 * 1024
        uploaded_files = []
        max_retries = 3
        retry_delay = 5  # seconds
        
        # Get total PDF size
        pdf_size = os.path.getsize(pdf_path)
        num_chunks = math.ceil(pdf_size / max_size_bytes)
        
        logger.info(f"Splitting PDF of size {pdf_size / (1024*1024):.2f}MB into {num_chunks} chunks")
        
        # Open the source PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            pages_per_chunk = math.ceil(total_pages / num_chunks)
            
            logger.info(f"Total pages: {total_pages}, Pages per chunk: {pages_per_chunk}")
            
            # Create chunks
            for chunk_num in range(num_chunks):
                start_page = chunk_num * pages_per_chunk
                end_page = min((chunk_num + 1) * pages_per_chunk, total_pages)
                
                # Create a new PDF writer
                writer = PyPDF2.PdfWriter()
                
                # Add pages to the writer
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])
                
                # Save the chunk to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    writer.write(temp_file)
                    temp_path = temp_file.name
                
                uploaded = False
                retries = 0
                last_error = None
                
                while not uploaded and retries < max_retries:
                    try:
                        # Upload the chunk
                        logger.info(f"Uploading chunk {chunk_num + 1}/{num_chunks} (pages {start_page + 1}-{end_page}), attempt {retries + 1}/{max_retries}")
                        with open(temp_path, 'rb') as f:
                            uploaded_file = genai.upload_file(f, mime_type="application/pdf")
                            uploaded_files.append(uploaded_file)
                            logger.info(f"Successfully uploaded chunk {chunk_num + 1}/{num_chunks}")
                            uploaded = True
                            
                            # Verify the upload by checking the file URI
                            if not hasattr(uploaded_file, 'uri') or not uploaded_file.uri:
                                raise ValueError("Upload verification failed: missing file URI")
                            
                    except Exception as e:
                        last_error = e
                        retries += 1
                        if retries < max_retries:
                            logger.warning(f"Upload attempt {retries} failed for chunk {chunk_num + 1}: {str(e)}")
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.error(f"Failed to upload chunk {chunk_num + 1} after {max_retries} attempts: {str(e)}")
                            raise Exception(f"Failed to upload chunk {chunk_num + 1}: {str(last_error)}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                # Add a small delay between chunks to avoid rate limits
                if chunk_num < num_chunks - 1:  # Don't delay after the last chunk
                    await asyncio.sleep(2)
        
        # Verify all chunks were uploaded
        if len(uploaded_files) != num_chunks:
            raise Exception(f"Upload verification failed: Expected {num_chunks} chunks but got {len(uploaded_files)}")
        
        logger.info(f"Successfully uploaded all {num_chunks} chunks")
        return uploaded_files

    @trace_llm_call("analyze_frames")
    async def _analyze_frames_and_update_summary(self, video: Video, frames_pdf_path: str, summary_json_path: str) -> Dict[str, Any]:
        """Analyze frames from PDF and update summary with best frames for each section."""
        try:
            # Get the model
            model = await self._get_video_analysis_model()
            
            # Load the summary
            with open(summary_json_path, 'r') as f:
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
                
                Instructions for frame analysis:
                1. For each frame in ALL PDF chunks:
                   - Look for stock chart graphs and analyze:
                     * Stock tickers visible in the chart
                     * Timeframe of the chart (daily, weekly, etc.)
                     * Technical analysis elements (trendlines, support/resistance, etc.)
                     * Visual quality and readability
                   - Record the full path of the frame from the PDF metadata
                   - Assign a quality score (0-10) based on clarity and information
                
                2. For each section in the summary:
                   - If the section discusses specific stocks:
                     * Find frames showing those exact stocks
                     * For each stock, select the highest quality frame
                     * Add the FULL FRAME PATHS to the section's frame_paths list
                   - If no specific stocks discussed, leave frame_paths as empty list
                
                3. CRITICAL: You MUST use the exact frame paths as shown in the PDF metadata.
                   DO NOT modify, shorten, or create new paths.
                
                4. Return your analysis in this exact JSON format:
                {{
                    "sections": [
                        {{
                            "topic": str,  # One of: "stock analysis", "general discussion", "market news", "market context", "trading context"
                            "stocks": [str],  # List of stock tickers
                            "start_time": float,
                            "end_time": float,
                            "summary": str,
                            "key_points": [str],
                            "overall_summary": str,
                            "frame_paths": [str]  # List of full paths to frames that best match this section
                        }}
                    ]
                }}
                
                Here is the existing summary to update with frames:
                {json.dumps(summary, indent=2)}
                
                Analyze ALL frames from ALL PDF chunks and return the complete updated summary following the schema above.
                Each section MUST have all fields from the original summary plus the frame_paths list.
                """
                
                parts = [{"text": prompt}]
                for pdf_file in pdf_files:
                    parts.append({"file_data": pdf_file})
                
                # Wait for rate limit before proceeding
                model_name = self.settings.model.transcription.name
                await self.model_manager._wait_for_rate_limit(model_name)
                
                try:
                    logger.info("Starting frame analysis with all PDF chunks...")
                    # Generate content with all parts
                    response = await asyncio.to_thread(
                        model.generate_content,
                        contents=parts,
                        generation_config={
                            'temperature': 0.1,
                            'top_p': 0.1,
                            'top_k': 1,
                            'max_output_tokens': 8192  # Ensure we have enough tokens for the response
                        }
                    )
                    logger.info("Frame analysis completed successfully")
                    
                    # Log raw response for debugging
                    response_text = response.text.strip()
                    logger.info(f"Raw frames analysis response: {response_text[:500]}...")
                    
                    # Try to parse the JSON response
                    try:
                        updated_summary = json.loads(response_text)
                    except json.JSONDecodeError:
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            json_str = json_match.group()
                            updated_summary = json.loads(json_str)
                        else:
                            error_msg = "No valid JSON found in frames analysis response"
                            logger.error(f"{error_msg}. Raw response: {response_text}")
                            raise ValueError(error_msg)
                    
                    # Validate the updated summary structure
                    if 'sections' not in updated_summary:
                        error_msg = "Updated summary missing required 'sections' field"
                        logger.error(f"{error_msg}. Keys found: {list(updated_summary.keys())}")
                        raise ValueError(error_msg)
                    
                    # Validate each section has the required fields including frame_paths
                    for section in updated_summary['sections']:
                        required_fields = ['topic', 'stocks', 'start_time', 'end_time', 
                                         'summary', 'key_points', 'overall_summary', 'frame_paths']
                        missing_fields = [field for field in required_fields if field not in section]
                        if missing_fields:
                            error_msg = f"Section missing required fields: {missing_fields}"
                            logger.error(f"{error_msg}. Section: {section}")
                            raise ValueError(error_msg)
                        
                        if not isinstance(section['frame_paths'], list):
                            error_msg = "frame_paths must be a list"
                            logger.error(f"{error_msg}. Type found: {type(section['frame_paths']).__name__}")
                            raise ValueError(error_msg)
                        
                        # Validate frame paths exist
                        for frame_path in section['frame_paths']:
                            if not os.path.exists(frame_path):
                                logger.warning(f"Frame path does not exist: {frame_path}")
                    
                    # Log frame analysis results
                    sections_count = len(updated_summary.get('sections', []))
                    total_frames = sum(len(s.get('frame_paths', [])) for s in updated_summary.get('sections', []))
                    sections_with_frames = sum(1 for s in updated_summary.get('sections', []) if s.get('frame_paths', []))
                    logger.info(f"Frame analysis results: {sections_count} sections, {total_frames} total frames, {sections_with_frames} sections with frames")
                    
                    # Save the updated summary
                    output_path = self.summaries_dir / f"{video.id}_analysis_with_frames.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(updated_summary, f, indent=2)
                    
                    logger.info(f"Saved updated summary with frames to {output_path}")
                    return updated_summary
                    
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        logger.warning(f"Rate limit hit during frame analysis: {str(e)}")
                        await asyncio.sleep(self.settings.rate_limit.retry_delay_seconds)
                        # Return original summary and let the caller retry if needed
                        return summary
                    else:
                        logger.error(f"Failed to process frames analysis response: {str(e)}", exc_info=True)
                        return summary
                    
            except Exception as e:
                logger.error(f"Failed to analyze frames and update summary: {str(e)}", exc_info=True)
                return summary

        except Exception as e:
            logger.error(f"Failed to analyze frames and update summary: {str(e)}")
            return summary

    async def _process_transcript(self, video: Video, raw_transcript: str) -> Dict[str, Any]:
        """Process raw transcript to identify stock discussions and their sections"""
        try:
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
            
            Return: {{
                'sections': list[Section]
            }}
            
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

            if self.langtrace is not None:
                @trace_llm_call("process_transcript")
                def analyze():
                    return model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.1,
                            'top_p': 0.1,
                            'top_k': 1
                        }
                    )
                response = analyze()
            else:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.1,
                        'top_p': 0.1,
                        'top_k': 1
                    }
                )

            try:
                # Log raw response for debugging
                response_text = response.text.strip()
                logger.info(f"Raw response from model: {response_text[:500]}...")  # Log first 500 chars
                
                # Try to find JSON in the response
                try:
                    # First try direct JSON parsing
                    parsed_response = json.loads(response_text)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from the text
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        json_str = json_match.group()
                        parsed_response = json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON found in response")
                
                logger.info(f"Successfully parsed response: {parsed_response}")
                
                # Validate response structure
                if 'sections' not in parsed_response:
                    raise ValueError("Response missing required 'sections' field")
                
                # Add metadata to the response
                current_date = datetime.now().strftime("%B %d %Y")
                final_response = {
                    "Date": current_date,
                    "Channel name": video.metadata.channel_name if video.metadata else "",
                    "Video name": video.metadata.video_title if video.metadata else "",
                    "sections": parsed_response.get("sections", [])
                }
                
                # Save both raw response and final JSON for debugging
                raw_response_path = self.summaries_dir / f"{video.id}_raw_llm_response.txt"
                with open(raw_response_path, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                
                json_response_path = self.summaries_dir / f"{video.id}_raw_llm.json"
                with open(json_response_path, 'w', encoding='utf-8') as f:
                    json.dump(final_response, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved raw response to {raw_response_path}")
                logger.info(f"Saved formatted JSON to {json_response_path}")
                
                # Create the return object
                overall_summary = "\n".join([
                    section.get("overall_summary", "")
                    for section in final_response.get("sections", [])
                ])
                
                # Ensure sections have frame_paths field
                for section in final_response.get("sections", []):
                    if "frame_paths" not in section:
                        section["frame_paths"] = []
                
                return {
                    "raw_response": overall_summary,
                    "stocks": final_response.get("sections", []),
                    "analysis_json": final_response
                }
            except Exception as e:
                logger.error(f"Failed to process model response: {str(e)}", exc_info=True)
                # Save the raw response for debugging
                error_path = self.summaries_dir / f"{video.id}_error_response.txt"
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {str(e)}\n\nRaw Response:\n{response_text}")
                logger.error(f"Saved error response to {error_path}")
                
                return self._create_default_analysis({
                    "transcript": raw_transcript,
                    "error": str(e)
                })
                
        except Exception as e:
            logger.error(f"Failed to process transcript: {str(e)}", exc_info=True)
            return self._create_default_analysis({
                "transcript": raw_transcript,
                "error": str(e)
            })

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
