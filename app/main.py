from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uuid
from typing import Optional, List, Dict
from pydantic import BaseModel, HttpUrl
import asyncio
import logging
import os

from app.models.video import Video, VideoSource, VideoStatus
from app.services.video_processor import VideoProcessor
from app.services.llm_service import LLMService
from app.services.vector_store import get_vector_store
from app.services.model_manager import ModelManager
from app.core.config import get_settings
from app.api.routes import router as api_router
from app.routes import settings_routes
from app.utils.langtrace_utils import init_langtrace, get_langtrace

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
    # Verify LangTrace instance
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
    allow_origins=["*"],  # In production, replace with your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Root route to serve index.html
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

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
                "extracting_frames": 0  # Add missing progress field
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
