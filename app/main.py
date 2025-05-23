import os
import json
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
from pydantic import BaseModel, HttpUrl

from app.models.video import Video, VideoSource, VideoStatus
from app.services.video_processor import VideoProcessor
from app.services.content_analyzer import ContentAnalyzer
from app.services.llm_service import LLMService
from app.services.vector_store import get_vector_store
from app.services.model_manager import ModelManager
from app.core.config import get_settings
from app.api.routes import router as api_router
from app.routes import settings_routes
from app.utils.langtrace_utils import init_langtrace, get_langtrace
from app.agents.technical_analysis_agent import TechnicalAnalysisAgent

# Configure logging at the root level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize LangTrace at startup
logger.info("Initializing LangTrace...")
if init_langtrace():
    logger.info("LangTrace initialized successfully")
    langtrace_instance = get_langtrace()
    if langtrace_instance:
        logger.info("LangTrace instance verified")
    else:
        logger.error("Failed to verify LangTrace instance")
else:
    logger.error("Failed to initialize LangTrace")

app = FastAPI(title=settings.app_name)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
logger.debug(f"Calculated static directory path: {static_dir}")

# Ensure static directory exists
if not os.path.exists(static_dir):
    logger.error(f"Static directory not found at: {static_dir}")
    raise RuntimeError(f"Static directory not found at: {static_dir}")
else:
    logger.info(f"Static directory found at: {static_dir}")
    # List contents of static directory
    files = os.listdir(static_dir)
    logger.info(f"Static directory contents: {files}")

# Mount static files directory BEFORE any routes
logger.info("Mounting static files directory...")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
logger.info("Static files directory mounted successfully")

# Configure templates
templates = Jinja2Templates(directory="app/templates")

# Create temp directory at startup
temp_dir = os.path.join("temp")
os.makedirs(temp_dir, exist_ok=True)
logger.info(f"Created temp directory at: {temp_dir}")

# Test route to verify server is working
@app.get("/api/test/ping")
async def test_ping():
    return {"status": "ok", "message": "Server is running"}

# Test page route
@app.get("/test")
async def test_page():
    """Serve the test page"""
    test_page_path = os.path.join(static_dir, "test.html")
    logger.info(f"Serving test.html from {test_page_path}")
    
    if not os.path.exists(test_page_path):
        logger.error(f"Test page not found at: {test_page_path}")
        raise HTTPException(status_code=404, detail="Test page not found")
        
    return FileResponse(test_page_path)

# Root route to serve index.html
@app.get("/")
async def root():
    index_page_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_page_path):
        logger.error(f"Index page not found at: {index_page_path}")
        raise HTTPException(status_code=404, detail="Index page not found")
        
    return FileResponse(index_page_path)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Include API routes
app.include_router(api_router)
app.include_router(settings_routes.router, prefix="/api")

@app.exception_handler(Exception)
async def handle_general_error(request: Request, exc: Exception):
    """Global error handler"""
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": str(exc)}
    )

# In-memory storage for video processing status
video_status: Dict[str, Dict] = {}

# Lazy service initialization
_video_processor = None
_llm_service = None
_vector_store = None
_content_analyzer = None

def get_video_processor():
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor(video_status=video_status)
    return _video_processor

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

def get_vector_store_service():
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store

def get_content_analyzer():
    global _content_analyzer
    if _content_analyzer is None:
        _content_analyzer = ContentAnalyzer()
    return _content_analyzer

async def process_video_background(video: Video):
    """Background task to process video"""
    try:
        logger.info(f"Starting background processing for video {video.id}")
        
        # Update status to PROCESSING
        video_status[video.id]["status"] = VideoStatus.PROCESSING
        logger.info(f"Updated status to PROCESSING for video {video.id}")
        
        # Get video processor instance
        video_processor = get_video_processor()
        
        # Process video
        processed_video = await video_processor.process_video(video)
        logger.info(f"Completed processing video {video.id}")
        
    except Exception as e:
        logger.error(f"Background task error for video {video.id}: {str(e)}", exc_info=True)
        video_status[video.id].update({
            "status": VideoStatus.FAILED,
            "error": str(e)
        })
        logger.info(f"Updated status to FAILED for video {video.id}")

class YouTubeRequest(BaseModel):
    url: HttpUrl

class PlaylistRequest(BaseModel):
    url: HttpUrl

class ChatRequest(BaseModel):
    query: str

@app.post("/analyze")
async def analyze_youtube_video(request: YouTubeRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"Received analyze request for URL: {request.url}")
        
        # Generate video ID
        video_id = str(uuid.uuid4())
        logger.info(f"Generated video ID: {video_id}")
        
        # Create video object with HttpUrl object
        video = Video(
            id=video_id,
            url=request.url,  # Pass the HttpUrl object directly
            source=VideoSource.YOUTUBE
        )
        logger.info(f"Created video object for {video_id}")
        
        # Initialize status
        video_status[video_id] = {
            "status": VideoStatus.PENDING,
            "error": None,
            "progress": {
                "downloading": 0,
                "transcribing": 0,
                "analyzing": 0,
                "extracting_frames": 0
            }
        }
        
        # Add background task
        background_tasks.add_task(process_video_background, video)
        
        return {"video_id": video_id}
        
    except Exception as e:
        logger.error(f"Error in analyze_youtube_video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{video_id}")
async def get_video_status(video_id: str):
    """Get the status of a video processing task"""
    if video_id not in video_status:
        raise HTTPException(status_code=404, detail="Video not found")
    return video_status[video_id]

@app.post("/analyze-playlist")
async def analyze_playlist(request: PlaylistRequest, background_tasks: BackgroundTasks):
    """Process a YouTube playlist"""
    try:
        logger.info(f"Received playlist analysis request for URL: {request.url}")
        
        # Get video processor instance
        video_processor = get_video_processor()
        
        # Extract video URLs from playlist
        video_urls = await video_processor.extract_playlist_urls(request.url)
        
        # Create video objects and start processing
        video_ids = []
        for url in video_urls:
            video_id = str(uuid.uuid4())
            video = Video(
                id=video_id,
                url=url,
                source=VideoSource.YOUTUBE
            )
            
            # Initialize status
            video_status[video_id] = {
                "status": VideoStatus.PENDING,
                "error": None,
                "progress": {
                    "downloading": 0,
                    "transcribing": 0,
                    "analyzing": 0
                }
            }
            
            video_ids.append(video_id)
            background_tasks.add_task(process_video_background, video)
        
        return {
            "playlist_id": str(uuid.uuid4()),
            "video_ids": video_ids,
            "total_videos": len(video_ids)
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_playlist: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the LLM about processed videos"""
    try:
        llm_service = get_llm_service()
        response = await llm_service.chat(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quota")
async def check_quota():
    """Check the remaining quota for the Gemini API key."""
    try:
        llm_service = get_llm_service()
        quota_info = await llm_service.model_manager.check_quota()
        return quota_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quota")
async def check_quota():
    """Check the remaining quota for the Gemini API key."""
    try:
        model_manager = ModelManager()
        quota_info = await model_manager.check_quota()
        return quota_info
    except Exception as e:
        logger.error(f"Failed to check quota: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test/analyze-pdf")
async def test_pdf_analysis(
    pdf_file: UploadFile = File(...),
    json_file: UploadFile = File(...),
    split_pdf: bool = Form(False)
):
    """Test endpoint to analyze a PDF file with frames using an existing JSON summary."""
    logger.info(f"Received analysis request - PDF: {pdf_file.filename}, JSON: {json_file.filename}, Split: {split_pdf}")
    
    try:
        # Save uploaded files temporarily
        temp_dir = os.path.join("temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        pdf_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.pdf")
        json_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.json")
        
        logger.info(f"Saving files to temp directory: {temp_dir}")
        
        try:
            # Save files
            with open(pdf_path, "wb") as f:
                content = await pdf_file.read()
                f.write(content)
                logger.info(f"Saved PDF file ({len(content)} bytes)")
            
            with open(json_path, "wb") as f:
                content = await json_file.read()
                f.write(content)
                logger.info(f"Saved JSON file ({len(content)} bytes)")
            
            # Read JSON content
            with open(json_path, "r") as f:
                json_content = json.load(f)
                logger.info("Successfully loaded JSON content")
            
            # Create a minimal Video object for the analysis
            video = Video(
                id=str(uuid.uuid4()),
                source=VideoSource.TEST,
                status=VideoStatus.PROCESSING
            )
            
            # Get content analyzer
            analyzer = get_content_analyzer()
            
            if split_pdf:
                # Use the new split PDF functionality
                logger.info("Using split PDF mode")
                result = await analyzer._analyze_frames_and_update_summary(
                    video=video,
                    frames_pdf_path=pdf_path,
                    summary_json_path=json_path
                )
            else:
                # Use direct file upload for smaller files
                logger.info("Using single file mode")
                try:
                    with open(pdf_path, 'rb') as f:
                        pdf_file = genai.upload_file(f, mime_type="application/pdf")
                    
                    logger.info(f"Successfully uploaded PDF file: {pdf_file.name}")
                    
                    model = await analyzer._get_video_analysis_model()
                    logger.info("Got video analysis model")
                    
                    # Create parts with correct schema
                    prompt = analyzer._get_frame_analysis_prompt(json_content)
                    logger.info("Generated analysis prompt")
                    
                    parts = [
                        {"text": prompt},
                        {"file_data": pdf_file}
                    ]
                    
                    # Generate content
                    logger.info("Generating content with Gemini...")
                    response = model.generate_content(contents=parts)
                    logger.info("Received response from Gemini")
                    
                    result = json.loads(response.text)
                    logger.info("Successfully parsed response JSON")
                    
                except Exception as e:
                    logger.error(f"Error in single file mode: {str(e)}", exc_info=True)
                    raise
            
            logger.info("Analysis completed successfully")
            return JSONResponse(content={"result": result})
            
        finally:
            # Clean up temporary files
            try:
                for temp_file in [pdf_path, json_path]:
                    if os.path.exists(temp_file):
                        os.chmod(temp_file, 0o666)  # Give write permission
                        os.remove(temp_file)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {str(e)}")
                
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in PDF analysis test: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/submit-report")
async def submit_report(
    report: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Handle manual report submission."""
    logger.info(f"Received report submission: {report.filename}")
    settings = get_settings()
    
    if not report.filename.endswith('.json'):
        logger.error(f"Invalid file type: {report.filename}")
        raise HTTPException(status_code=400, detail="Only JSON files are accepted")
        
    try:
        # Read content
        content = await report.read()
        logger.debug(f"Report size: {len(content)} bytes")
        
        try:
            # Parse JSON just to validate format
            report_data = json.loads(content)
            logger.info("Report parsed successfully")
            
            # Create temp directory if it doesn't exist
            temp_dir = Path(settings.storage.temp_dir)
            if not temp_dir.is_absolute():
                # If relative path, make it absolute from project root
                project_root = Path(__file__).parent.parent
                temp_dir = project_root / temp_dir
            temp_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Using temp directory: {temp_dir}")
            logger.debug(f"Temp directory exists: {temp_dir.exists()}")
            logger.debug(f"Temp directory is absolute: {temp_dir.is_absolute()}")
            
            # Save uploaded file temporarily
            tmp_path = temp_dir / f"{uuid.uuid4()}_report.json"
            # Write file with explicit encoding
            tmp_path.write_text(json.dumps(report_data, indent=2), encoding='utf-8')
            logger.debug(f"Saved report to temporary file: {tmp_path}")
            logger.debug(f"File exists after write: {tmp_path.exists()}")
            if tmp_path.exists():
                logger.debug(f"File size after write: {tmp_path.stat().st_size} bytes")
                logger.debug(f"File contents: {tmp_path.read_text(encoding='utf-8')[:100]}...")
            
            try:
                # Validate Notion settings
                settings = get_settings()
                if not settings.notion.api_key:
                    raise ValueError("Notion API token not found in settings. Please check your .env file for NOTION_API_KEY.")
                    
                if not settings.notion.database_id:
                    raise ValueError("Notion database ID not found in settings. Please check your .env file for NOTION_DATABASE_ID.")
                
                # Log credentials being used (mask sensitive parts)
                api_key = settings.notion.api_key
                db_id = settings.notion.database_id
                logger.info(f"Using Notion API key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
                logger.info(f"Using Notion database ID: {db_id}")
                    
                # Validate API token first
                from notion_client import Client
                notion = Client(auth=settings.notion.api_key)
                try:
                    # Try a simple API call to validate token
                    notion.users.me()
                    logger.info("Notion API token validated successfully")
                except Exception as e:
                    raise ValueError(f"Invalid Notion API token. Please check NOTION_API_KEY in your .env file. Error: {str(e)}")
                
                # Now validate database access
                try:
                    db = notion.databases.retrieve(database_id=settings.notion.database_id)
                    logger.info(f"Found Notion database: {db['title'][0]['text']['content']}")
                except Exception as e:
                    raise ValueError(f"Could not access Notion database. Please check NOTION_DATABASE_ID in .env file. Error: {str(e)}")
                
                # Create and execute technical analysis agent
                agent = TechnicalAnalysisAgent()
                logger.info("Starting technical analysis agent execution")
                
                # Convert to string and check if file exists
                tmp_path_str = str(tmp_path)
                logger.debug(f"Passing file path to agent: {tmp_path_str}")
                logger.debug(f"File exists before agent: {Path(tmp_path_str).exists()}")
                
                if background_tasks:
                    # Run in background if background_tasks is provided
                    background_tasks.add_task(agent.execute, tmp_path_str)
                    logger.info("Technical analysis scheduled for background execution")
                    cleanup_message = "Report scheduled for processing"
                else:
                    # Run synchronously
                    await agent.execute(tmp_path_str)
                    logger.info("Technical analysis completed successfully")
                    cleanup_message = "Report processed successfully"
                    
                return {
                    "status": "success",
                    "message": cleanup_message,
                    "report_size": len(content)
                }
                
            except Exception as e:
                logger.error(f"Error during technical analysis: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")
            finally:
                # Keep the file for debugging
                logger.debug(f"Keeping temporary file for debugging: {tmp_path}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error in submit_report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
