import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
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

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)
settings = get_settings()

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
        is_truncated = finish_reason == 'FINISH_REASON_MAX_TOKENS' or finish_reason == 2
        
        if is_truncated:
            logger.info("Response was truncated due to token limit (FINISH_REASON_MAX_TOKENS)")
        
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
    additional_contents: List[Dict[str, Any]] = None
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
            "Continue from where you have stopped in the following provided response, "
            "add only the missing data, exactly from where you have stopped, "
            "don't add any greetings or extra messages, follow the original prompt\n\n"
            f"Original prompt:\n{original_prompt}\n\n"
            f"Previous truncated response:\n{combined_text}"
        )
        
        try:
            # Prepare contents for continuation
            contents = [{"text": continuation_prompt}]
            
            # Add any additional file data or other contents
            if additional_contents:
                file_count = sum(1 for item in additional_contents if "file_data" in item)
                print(f"{BLUE}ℹ Including {file_count} files in continuation request {continuation_attempts}:{RESET}")
                
                # Display file information
                for idx, item in enumerate(additional_contents):
                    if "file_data" in item and hasattr(item["file_data"], "display_name"):
                        file_name = item["file_data"].display_name
                        file_uri = getattr(item["file_data"], "uri", "unknown_uri")
                        print(f"{BLUE}  - File {idx+1}: {file_name} (URI: {file_uri[:30]}...){RESET}")
                
                contents.extend(additional_contents)
                logger.info(f"Including {len(additional_contents)} additional content items in continuation request")
            else:
                print(f"{YELLOW}⚠ No additional file data included in continuation request!{RESET}")
            
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

    async def _split_pdf_and_upload(self, pdf_path: str) -> List[Any]:
        """Split a PDF into smaller chunks for processing and upload each chunk."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)
                    
                    # Define chunk size and number of chunks
                    chunk_size = 40  # Pages per chunk
                    num_chunks = (total_pages + chunk_size - 1) // chunk_size
                    
                    logger.info(f"Splitting PDF with {total_pages} pages into {num_chunks} chunks")
                    
                    # Split into chunks
                    chunk_files = []
                    for i in range(num_chunks):
                        pdf_writer = PyPDF2.PdfWriter()
                        start_page = i * chunk_size
                        end_page = min((i + 1) * chunk_size, total_pages)
                        
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
                            print(f"{GREEN}✓ File '{file.display_name}' ({os.path.basename(chunk_path)}) is now ACTIVE and ready to use. URI: {file.uri[:30]}...{RESET}")
                            logger.info(f"Uploaded chunk {chunk_path}")
                        except Exception as e:
                            print(f"{YELLOW}✗ Failed to upload chunk {os.path.basename(chunk_path)}: {str(e)}{RESET}")
                            logger.error(f"Failed to upload chunk {chunk_path}: {str(e)}")
                            continue
                    
                    print(f"{GREEN}Total files uploaded: {len(uploaded_files)}/{len(chunk_files)}{RESET}")
                    return uploaded_files
                    
        except Exception as e:
            print(f"{YELLOW}✗ Failed to split and upload PDF: {str(e)}{RESET}")
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
                print(f"{GREEN}Successfully uploaded {len(pdf_files)} PDF chunks for frame analysis{RESET}")
                logger.info(f"Successfully uploaded {len(pdf_files)} PDF chunks")
                
                # List all uploaded files with details
                print(f"{BLUE}=== PDF Files Available for Analysis ==={RESET}")
                for idx, pdf_file in enumerate(pdf_files):
                    file_name = getattr(pdf_file, "display_name", f"file_{idx}")
                    file_uri = getattr(pdf_file, "uri", "unknown_uri")
                    file_state = getattr(pdf_file, "state", "unknown_state")
                    print(f"{BLUE}  {idx+1}. {file_name} - State: {file_state} - URI: {file_uri[:30]}...{RESET}")
                print(f"{BLUE}======================================={RESET}")
                
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
                   - Do not add irrelevant images that doesn't show a stock chart
            
                2. For each section in the summary:
                   - If the section discusses specific stocks:
                     * Find frames showing those exact stocks
                     * For each stock, select the highest quality frame
                     * Add the FULL FRAME PATHS to the section's frame_paths list
                   - If no specific stocks discussed or there is no visible chart, leave frame_paths as empty list
            
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
                    print(f"{BLUE}Starting frame analysis with {len(pdf_files)} PDF chunks...{RESET}")
                    logger.info("Starting frame analysis with all PDF chunks...")
                    # Generate content with all parts
                    generation_config = {
                        'temperature': 0.1,
                        'top_p': 0.1,
                        'top_k': 40
                    }
                    
                    # Prepare PDF file contents
                    pdf_contents = [{"file_data": pdf_file} for pdf_file in pdf_files]
                    
                    # Log detail about the content being sent
                    print(f"{BLUE}Sending initial request with prompt ({len(prompt)} chars) and {len(pdf_contents)} PDF files{RESET}")
                    
                    response = await asyncio.to_thread(
                        model.generate_content,
                        contents=[{"text": prompt}] + pdf_contents,
                        generation_config=generation_config
                    )
                    
                    # Check initial response finish reason
                    finish_reason = None
                    if hasattr(response, "candidates") and response.candidates:
                        finish_reason = response.candidates[0].finish_reason
                        print(f"{BLUE}Initial response finish reason: {finish_reason}{RESET}")
                    
                    print(f"{GREEN}Initial frame analysis completed. Response length: {len(response.text)} chars{RESET}")
                    logger.info("Initial frame analysis completed")
                    
                    # Handle potential chunked response - provide PDF files for continuations too
                    response_text = await handle_chunked_response(
                        model=model,
                        original_prompt=prompt,
                        first_response=response,
                        content_generator_kwargs={'generation_config': generation_config},
                        additional_contents=pdf_contents
                    )
                    
                    # Save raw response
                    raw_response_path = self.summaries_dir / f"{video.id}_raw_frames_analysis.txt"
                    with open(raw_response_path, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    print(f"{GREEN}Saved raw frames analysis ({len(response_text)} chars) to {raw_response_path}{RESET}")
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
            1. Break down the transcript into sections based on the topics listed above
            2. For each section provide all fields defined in the schema
            3. When analyzing stock-specific sections:
               - CRITICAL: Create a SEPARATE section for EACH individual stock ticker/name
               - Each "stock analysis" section MUST have EXACTLY ONE stock in the 'stocks' array
               - For example, if AAPL, MSFT, and AMZN are discussed sequentially, create THREE separate sections
               - Include specific times when each individual stock is discussed
               - Note any market data or metrics mentioned for that specific stock
               - Highlight significant trends or patterns for that specific stock
            4. Important:
               - Skip any non-relevant discussions, commercials, or advertising
               - Focus on actionable trading insights and market analysis
               - Be specific and detailed in your summaries
               - Use proper stock ticker symbols
               - DO NOT combine multiple stocks in a single section
            
            IMPORTANT: 
            - Your response must be a valid JSON array matching the schema above.
            - Do not include any text before or after the JSON.
            - Make sure that you complete the JSON array with proper closing brackets.
            - Start your response with [
            - End your response with ]
            
            Here is the transcript to analyze:
            {raw_transcript}
            """

            # Generation configuration
            generation_config = {
                'temperature': 0.1,
                'top_p': 0.1,
                'top_k': 40
            }

            # Generate content
            response = await asyncio.to_thread(
                model.generate_content,
                contents=[{"text": prompt}],
                generation_config=generation_config
            )
            
            # Handle potential chunked response
            response_text = await handle_chunked_response(
                model=model,
                original_prompt=prompt,
                first_response=response,
                content_generator_kwargs={'generation_config': generation_config}
            )
            
            # Log partial response for debugging
            logger.info(f"Response first 500 chars: {response_text[:500]}...")
            
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
            # Check if sections is stored as a string and convert it to proper JSON object
            if 'sections' in analysis and isinstance(analysis['sections'], str):
                try:
                    # Try to parse the sections string as JSON
                    logger.info("Converting 'sections' from string to JSON object before saving")
                    analysis['sections'] = json.loads(analysis['sections'])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse 'sections' as JSON during initial save: {str(e)}")
            
            # Save the analysis JSON
            output_path = self.summaries_dir / f"{video.id}_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved analysis to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            raise


    def _clean_json_response(self, response_text: str) -> str:
        """Clean and prepare JSON response for parsing."""
        try:
            # First, remove any ```json and ``` markers
            # Find content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            
            # Try to extract content between markdown code blocks first
            start_idx = response_text.find(start_marker)
            if start_idx != -1:
                # Move past the start marker
                start_idx += len(start_marker)
                end_idx = response_text.find(end_marker, start_idx)
                if end_idx != -1:
                    response_text = response_text[start_idx:end_idx].strip()
            
            # If we still have markdown code blocks, do more aggressive cleanup
            if "```" in response_text:
                # Strip all lines with code block markers
                lines = response_text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```") and not line.strip().endswith("```")]
                response_text = "\n".join(lines).strip()
            
            # Clean any remaining code block markers that might be inline
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Replace invalid control characters
            # This regex matches all ASCII control characters except common whitespace (\t, \n, \r)
            import re
            response_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response_text)
            
            # Make sure we have valid JSON 
            # Try to parse it to find errors, but don't actually use the parsed result yet
            try:
                json.loads(response_text)
            except json.JSONDecodeError as e:
                # Log the error location for debugging
                logger.warning(f"JSON validation failed at char {e.pos}: {e.msg}")
                # Attempt to fix common issues with the JSON
                if "control character" in str(e):
                    # Convert the string to bytes, decode with errors='ignore' to remove invalid chars
                    response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                
                # Try once more to validate
                try:
                    json.loads(response_text)
                    logger.info("JSON was fixed successfully after cleaning")
                except json.JSONDecodeError:
                    logger.warning("JSON still invalid after cleaning")
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {str(e)}")
            logger.debug(f"Problematic response text: {response_text[:200]}...")
            return response_text
