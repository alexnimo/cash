import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from app.services.json_utils import lint_json, fix_json, clean_json_response
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from datetime import datetime, timedelta
from app.core.config import get_settings
from app.models.video import Video, VideoStatus
from app.services.model_manager import ModelManager
from app.services.report_generator import ReportGenerator
from app.utils.langtrace_utils import get_langtrace, trace_llm_call, init_langtrace
from app.utils.path_utils import get_storage_subdir, get_storage_path
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
RED = "\033[91m"  # Added missing RED color
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
                file_count = sum(1 for item in additional_contents if "file_data" in item)
                print(f"{BLUE}ℹ Including {file_count} files in continuation request {continuation_attempts}:{RESET}")
                
                # Display file information
                for idx, item in enumerate(additional_contents):
                    if "file_data" in item and hasattr(item["file_data"], "display_name"):
                        file_name = item["file_data"].display_name
                        file_uri = getattr(item["file_data"], "uri", "unknown_uri")
                        print(f"{BLUE}  - File {idx+1}: {file_name} (URI: {file_uri[:30]}...){RESET}")
                print(f"{BLUE}======================================={RESET}")
                
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
            
            # Check if we're dealing with JSON content structure
            is_json_content = (
                (combined_text.strip().startswith('[') or combined_text.strip().startswith('{')) and 
                ('"' in combined_text or "'" in combined_text)  # Simple heuristic for JSON-like content
            )
            
            if is_json_content:
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
    
    # Before returning, try to validate and fix the JSON if it's likely to be a JSON structure
    if combined_text.strip().startswith('[') and combined_text.strip().endswith(']'):
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

class ContentAnalyzer:
    def __init__(self):
        """Initialize the ContentAnalyzer."""
        self.settings = get_settings()
        self.model_manager = ModelManager(self.settings)
        
        # Initialize model as None (it will be lazily loaded)
        self.video_analysis_model = None
        
        # Use the unified storage path utilities
        self.video_dir = get_storage_subdir("videos")
        self.transcript_dir = get_storage_subdir("videos/transcripts")
        self.raw_transcript_dir = get_storage_subdir("videos/raw_transcripts")
        self.summaries_dir = get_storage_subdir("videos/summaries")
        self.reports_dir = get_storage_subdir("videos/reports")
        
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
                   - You can pick one additional frame for eacch stock in case you can cleary recognize
                    Time frame. In most cases time frames can be recognized by the chart title (w , weekly, m, monthly) or the timeframe label.
                   - Do not add irrelevant images that doesn't show a stock chart
            
                2. For each section in the summary:
                   - If the section discusses specific stocks:
                     * Find frames showing those exact stocks
                     * For each stock, select the highest quality frame
                     * Add the FULL FRAME PATHS to the section's frame_paths list
                   - If no specific stocks discussed or there is no visible chart, leave frame_paths as empty list
                   - Do not add any general frames which do not show a stock chart
            
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
                
                print(f"{BLUE}Starting frame analysis with {len(pdf_files)} PDF chunks...{RESET}")
                logger.info("Starting frame analysis with all PDF chunks...")
                
                # Generation configuration
                generation_config = {
                    'temperature': 0.1,
                    'top_p': 0.1,
                    'top_k': 40,
                    'response_mime_type': 'application/json'  # Force JSON response format
                }
                
                # Prepare PDF file contents
                pdf_contents = [{"file_data": pdf_file} for pdf_file in pdf_files]
                
                # Log detail about the content being sent
                print(f"{BLUE}Sending initial request with prompt ({len(prompt)} chars) and {len(pdf_contents)} PDF files{RESET}")
                
                try:
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
                    
                    # Direct JSON parsing - minimal approach
                    try:
                        # Parse JSON directly - no sanitization or validation
                        sections = json.loads(response_text)
                        
                        # Save the complete frame analysis with all details
                        frame_analysis_path = self.summaries_dir / f"{video.id}_frames_analysis.json"
                        with open(frame_analysis_path, 'w', encoding='utf-8') as f:
                            json.dump(sections, f, indent=2)
                        print(f"{GREEN}Saved complete frames analysis to {frame_analysis_path}{RESET}")
                        logger.info(f"Saved complete frames analysis to {frame_analysis_path}")
                        
                        # Ensure the sections include frame_paths before storing
                        for section in sections:
                            if 'frame_paths' not in section:
                                section['frame_paths'] = []
                        
                        # Store the parsed sections directly
                        summary["sections"] = sections
                        return summary
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse frame analysis JSON: {str(e)}")
                        # Store raw response without attempting complex recovery
                        summary["sections"] = response_text  # Store raw response as fallback
                        return summary
                    
                except Exception as e:
                    logger.error(f"Failed to generate or process frame analysis: {str(e)}")
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
                'top_k': 40,
                'response_mime_type': 'application/json'  # Force JSON response format
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
            
            # Clean the response text of any markdown or unwanted artifacts
            logger.info("Cleaning response of any markdown or other artifacts")
            cleaned_response = clean_json_response(response_text)
            
            # Log partial response for debugging
            logger.info(f"Response first 500 chars: {cleaned_response[:500]}...")
            
            # Save raw and cleaned responses
            raw_response_path = self.summaries_dir / f"{video.id}_raw_response.txt"
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.info(f"Saved raw response to {raw_response_path}")
            
            cleaned_response_path = self.summaries_dir / f"{video.id}_cleaned_response.json"
            with open(cleaned_response_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_response)
            logger.info(f"Saved cleaned response to {cleaned_response_path}")
            
            # Use our JSON linter and fixer for robust parsing
            logger.info("Validating and potentially fixing JSON response...")
            
            # Important: Instead of parsing the JSON ourselves, just treat everything as a string
            # and pass it directly to the output structure. This preserves the actual content
            # even if it has JSON issues.
            
            # First try direct JSON parsing just to see if it works
            try:
                # Try parsing - this is just to check validity
                json.loads(cleaned_response)
                logger.info("JSON response is valid on first attempt")
                # But we won't use this parsed object directly
            except json.JSONDecodeError as e:
                # JSON parsing failed, use our linter to identify issues
                logger.warning(f"JSON parsing failed: {str(e)}")
                print(f"\033[93m⚠️ \033[91mWARNING: JSON PARSING FAILED - continuing with original structure\033[0m")
                
                # Try to fix the JSON - optional step
                try:
                    fixed_json, fixes_applied = fix_json(cleaned_response)
                    if fixed_json:
                        logger.info(f"Successfully fixed JSON. Fixes applied: {len(fixes_applied)}")
                        # Save the fixed JSON as the cleaned response
                        cleaned_response = fixed_json
                        # Save for debugging
                        fixed_path = self.summaries_dir / f"{video.id}_fixed_response.json"
                        with open(fixed_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_json)
                except Exception as fix_error:
                    logger.warning(f"Error during JSON fixing: {str(fix_error)}")
            
            # The key insight: Don't try to parse the JSON again, just use the raw string
            # which already has the structure we want, even if it's not perfectly valid JSON
            
            # We'll trick the system by creating a pre-structure with the cleaned response
            # embedded in it as the 'sections' field directly
            sections = cleaned_response  # This is a string, not a list!
            
            # Create the analysis object with special handling for sections
            # If sections is already a list (parsed successfully), use it directly
            # If it's a string (the cleaned_response), we need to save it differently
            if isinstance(sections, str):
                # Special handling for string sections - we'll save a modified structure
                # that will preserve the JSON structure in the output file
                sections_str = sections
                try:
                    # Try to make the string into a proper JSON array if it isn't one already
                    if not sections_str.strip().startswith('['):
                        sections_str = f"[{sections_str}]"
                    if not sections_str.strip().endswith(']'):
                        sections_str = f"{sections_str}]"
                    
                    # We'll still try to create a proper JSON structure for the complete file
                    # This builds a JSON object as a string, with the sections_str embedded directly
                    raw_json = f'''{{
  "Date": "{datetime.now().strftime('%B %d %Y')}",
  "Channel name": "{video.metadata.channel_name if video.metadata else ''}",
  "Video name": "{video.metadata.video_title if video.metadata else ''}",
  "sections": {sections_str}
}}'''
                    
                    # Save this directly without parsing
                    direct_path = self.summaries_dir / f"{video.id}_direct_analysis.json"
                    with open(direct_path, 'w', encoding='utf-8') as f:
                        f.write(raw_json)
                    logger.info(f"Saved direct analysis containing original sections structure to {direct_path}")
                    
                    # For the return value, we still need a Python dict
                    # Try to parse it, but if it fails, use a minimal structure
                    try:
                        analysis_json = json.loads(raw_json)
                        logger.info("Successfully parsed combined analysis JSON")
                    except json.JSONDecodeError:
                        # If that fails too, create a fallback structure that at least has the metadata
                        analysis_json = {
                            "Date": datetime.now().strftime("%B %d %Y"),
                            "Channel name": video.metadata.channel_name if video.metadata else "",
                            "Video name": video.metadata.video_title if video.metadata else "",
                            "sections": []
                        }
                except Exception as e:
                    logger.error(f"Error creating direct analysis: {str(e)}")
                    # Fallback to standard structure
                    analysis_json = {
                        "Date": datetime.now().strftime("%B %d %Y"),
                        "Channel name": video.metadata.channel_name if video.metadata else "",
                        "Video name": video.metadata.video_title if video.metadata else "",
                        "sections": []
                    }
            else:
                # Normal case: sections is already a list
                analysis_json = {
                    "Date": datetime.now().strftime("%B %d %Y"),
                    "Channel name": video.metadata.channel_name if video.metadata else "",
                    "Video name": video.metadata.video_title if video.metadata else "",
                    "sections": sections
                }
            
            # Create overall summary safely
            try:
                if isinstance(sections, list):
                    overall_summary = "\n".join([
                        section.get("overall_summary", "")
                        for section in sections
                    ])
                else:
                    overall_summary = "Analysis summary"  # Fallback
            except Exception as e:
                logger.warning(f"Error creating overall summary: {str(e)}")
                overall_summary = "Analysis summary"  # Fallback
            
            # Return the full analysis
            return {
                "raw_response": overall_summary,
                "stocks": sections,  # Keep original sections for compatibility
                "analysis_json": analysis_json
            }
                
        except Exception as e:
            logger.exception(f"Error in transcript processing: {str(e)}")
            print(f"\033[93m⚠️ \033[91mCRITICAL ERROR: Transcript processing failed - {str(e)}\033[0m")
            
            # Even on exception, create a valid structure to maintain compatibility
            sections = [{
                "topic": "processing error",
                "stocks": [],
                "start_time": 0.0,
                "end_time": 0.0,
                "summary": "Unable to process transcript",
                "key_points": ["Processing encountered an error"],
                "overall_summary": f"Processing error encountered. Raw transcript length: {len(raw_transcript)} characters"
            }]
            
            # Create the analysis object with standard structure despite error
            analysis_json = {
                "Date": datetime.now().strftime("%B %d %Y"),
                "Channel name": video.metadata.channel_name if video.metadata else "",
                "Video name": video.metadata.video_title if video.metadata else "",
                "sections": sections
            }
            
            # Save error for debugging
            error_path = self.summaries_dir / f"{video.id}_transcript_processing_error.txt"
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}\n\nTraceback information saved in logs")
            logger.error(f"Saved error info to {error_path} (processing error)")
            
            # Return standard structure despite error
            return {
                "raw_response": "Processing error",
                "stocks": sections,
                "analysis_json": analysis_json
            }


    
    async def _save_combined_analysis(self, video: Video, analysis: Dict[str, Any]) -> str:
        """Save all analysis data in a single JSON file."""
        try:
            # Ensure analysis is properly structured for saving
            if 'analysis_json' in analysis and isinstance(analysis['analysis_json'], dict):
                # Use the pre-built analysis_json object directly
                output_analysis = analysis['analysis_json']
                logger.info("Using pre-structured analysis_json for saving")
            else:
                # Create a properly structured analysis object
                logger.info("Creating structured analysis object for saving")
                
                # Check if sections is stored as a string and convert it to proper JSON object
                sections = analysis.get('stocks', [])
                if isinstance(sections, str):
                    try:
                        # Simply parse the string to JSON - it should already be valid JSON with response_mime_type
                        logger.info("Converting 'sections' from string to JSON object before saving")
                        sections = json.loads(sections)
                        logger.info("Successfully converted 'sections' string to JSON object")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse 'sections' as JSON during save: {str(e)}")
                        print(f"\033[93m⚠️ \033[91mWARNING: Sections parse error - Using raw text in analysis\033[0m")
                        
                        # Instead of failing, use the raw string in a synthetic section
                        sections = [{
                            "topic": "raw content",
                            "stocks": [],
                            "start_time": 0.0,
                            "end_time": 0.0,
                            "summary": "Content from unparsed sections",
                            "key_points": ["Using raw text due to parsing error"],
                            "overall_summary": sections[:1000] + "..." if len(sections) > 1000 else sections
                        }]
                        
                        # Save the problematic string for debugging
                        error_path = self.summaries_dir / f"{video.id}_sections_parse_error.txt"
                        with open(error_path, 'w', encoding='utf-8') as f:
                            f.write(f"Error: {str(e)}\n\nOriginal sections string:\n{sections}")
                        logger.error(f"Saved sections parse error to {error_path}")
                
                # Create the output analysis structure - maintain expected format
                output_analysis = {
                    "Date": datetime.now().strftime("%B %d %Y"),
                    "Channel name": video.metadata.channel_name if video.metadata else "",
                    "Video name": video.metadata.video_title if video.metadata else "",
                    "sections": sections
                }
            
            # Save the analysis JSON 
            output_path = self.summaries_dir / f"{video.id}_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_analysis, f, indent=2, ensure_ascii=False)
            
            # Prepare for the final analysis file - validate JSON first
            # Convert to string with indentation for linting and fixing
            json_string = json.dumps(output_analysis, indent=2, ensure_ascii=False)
            
            # Check if the JSON is valid (this should always be true at this point, but check anyway)
            error_info = lint_json(json_string)
            if error_info:
                logger.warning(f"Found JSON issues in final analysis: {error_info['readable_error']}")
                logger.info(f"Attempting to fix JSON issues before saving final analysis...")
                print(f"\033[93m⚠️ \033[91mWARNING: JSON issues in final analysis - attempting fix\033[0m")
                
                # Try to fix JSON
                fixed_json, fixes_applied = fix_json(json_string)
                if fixed_json:
                    logger.info(f"Successfully fixed JSON issues. Fixes applied: {len(fixes_applied)}")
                    for fix in fixes_applied:
                        logger.debug(f"JSON Fix: {fix}")
                    
                    # Parse the fixed JSON back to a Python object
                    try:
                        output_analysis = json.loads(fixed_json)
                        logger.info("Successfully parsed fixed JSON back to Python object")
                    except json.JSONDecodeError as e:
                        # Even if parsing fails, continue with original structure
                        # Just report the error but don't alter the output_analysis
                        logger.error(f"Failed to parse fixed JSON: {str(e)}")
                        print(f"\033[93m⚠️ \033[91mWARNING: Couldn't parse fixed JSON - continuing with original structure\033[0m")
                        
                        # Save the problematic fixed string for debugging
                        error_path = self.summaries_dir / f"{video.id}_fixed_json_parse_error.txt"
                        with open(error_path, 'w', encoding='utf-8') as f:
                            f.write(f"Error: {str(e)}\n\nFixed JSON string:\n{fixed_json}")
                        logger.error(f"Saved fixed JSON parse error to {error_path}")
                else:
                    logger.warning(f"Could not fix JSON issues. Fixes attempted: {fixes_applied}")
                    print(f"\033[93m⚠️ \033[91mWARNING: Could not fix JSON issues - continuing with original structure\033[0m")
            else:
                logger.info("Final analysis JSON is valid, no fixes needed")
                
            # Save the final analysis file
            final_output_path = self.summaries_dir / f"{video.id}_final_analysis.json"
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved analysis to {output_path} and {final_output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            raise

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
