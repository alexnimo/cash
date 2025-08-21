"""Module for loading and managing application settings."""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache
from pydantic import BaseModel, Field
import yaml
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Load environment variables at module level
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path.absolute()}")
    logger.info(f"FREEIMAGE_API_KEY present: {bool(os.getenv('FREEIMAGE_API_KEY'))}")
else:
    logger.warning(f"No .env file found at {env_path.absolute()}")

print("Settings module loaded")

class VideoStorageSettings(BaseModel):
    base_dir: str = Field(default="videos")

class LoggingSettings(BaseModel):
    level: str = Field(default="INFO")

class LangTraceSettings(BaseModel):
    enabled: bool = Field(default=True)
    api_key: str = Field(default="")
    project_id: str = Field(default="vid-expert")
    trace_dir: str = Field(default="./traces")
    sampling_rate: float = Field(default=1.0)
    trace_models: bool = Field(default=True)

class ModelSettings(BaseModel):
    type: str
    name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    enabled: Optional[bool] = True
    dimension: Optional[int] = None

class ModelConfig(BaseModel):
    video_analysis: ModelSettings
    frame_analysis: ModelSettings
    transcription: ModelSettings
    embedding: Optional[ModelSettings] = None

class VectorStoreConfig(BaseModel):
    type: str
    config: Dict[str, Any]
    metadata_config: Dict[str, List[str]] = Field(
        default={"indexed": ["video_id", "timestamp", "content_type", "language"]}
    )

class StorageConfig(BaseModel):
    base_path: str = Field(default="./data")  # Base storage location for all application data
    # This is the only required setting; subdirectories will be created automatically

class APIConfig(BaseModel):
    gemini_api_key: str
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: List[str] = Field(default=["*"])
    rate_limit_window: int = Field(default=60)
    max_requests: int = Field(default=60)
    gemini_rpm: int = Field(default=12)  # Default to 12 RPM (5 second interval)
    gemini_rpm_override: Optional[int] = None  # For UI override

class ProcessingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    max_tokens_per_request: int
    batch_size: int
    timeout: int
    max_parallel_chunks: int
    max_video_duration: int

class LLMConfig(BaseModel):
    type: str = Field(default="gemini")
    name: str = Field(default="gemini-pro")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None  # Add if needed
    # Add other relevant LLM settings if they exist in your config under agents.llm

class AgentPrompts(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    user_prompt_template: str = "{input}"
    # Add other prompt fields if they exist

class AgentEmbeddingConfig(BaseModel):
    provider: str
    model: str 
    dimension: int
    metric: str
    metadata_fields: Optional[List[str]] = None

class AgentPineconeConfig(BaseModel):
    index_name: str
    cloud: str
    region: str
    environment: Optional[str] = None 

class AgentConfig(BaseModel):
    enabled: bool = True 
    agent_debug: bool = True 
    debug_dir: str = "debug" 
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: AgentPrompts = Field(default_factory=AgentPrompts)
    embedding: AgentEmbeddingConfig 
    pinecone: AgentPineconeConfig  
    rag: Optional[Dict[str, Any]] = None 

class NotionConfig(BaseModel):
    api_key: str
    database_id: str
    parent_page_id: Optional[str] = None
    stock_ticker_property: str = Field(default="Stock Ticker")
    charts_property: str = Field(default="Charts")
    ta_summary_property: str = Field(default="TA Summary")

class FreeimageSettings(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("FREEIMAGE_API_KEY", ""))

class TranscriptGenerationSettings(BaseModel):
    max_chunk_duration_minutes: int = Field(default=30)

class JanitorConfig(BaseModel):
    """Configuration for the janitor (file cleanup) service"""
    enabled: bool = Field(default=True)
    schedule: str = Field(default="0 1 * * *")  # Daily at 1:00 AM
    cleanup_paths: List[str] = Field(default=["videos", "temp", "data/summaries", "traces"])
    file_patterns: List[str] = Field(default=["*.mp4", "*.wav", "*.txt", "*.json"])
    retention_hours: int = Field(default=168)  # 7 days
    dry_run: bool = Field(default=False)
    exclude_patterns: List[str] = Field(default=[])  # Files to exclude from cleanup
    log_deletions: bool = Field(default=True)  # Log each file deletion
    preserve_recent_files: bool = Field(default=True)  # Keep files from last 24h regardless of retention

class Settings(BaseModel):
    app_name: str = "Video Analyzer"
    model: ModelConfig
    vector_store: VectorStoreConfig
    storage: StorageConfig
    api: APIConfig
    langtrace: LangTraceSettings
    processing: ProcessingConfig
    logging: LoggingSettings
    video_storage: VideoStorageSettings
    agents: AgentConfig = Field(default_factory=AgentConfig)
    notion: NotionConfig
    freeimage: FreeimageSettings = Field(default_factory=FreeimageSettings)
    transcript_generation: TranscriptGenerationSettings = Field(default_factory=TranscriptGenerationSettings)
    janitor: JanitorConfig = Field(default_factory=JanitorConfig)

_settings_instance = None

def load_settings() -> Settings:
    """Load settings from config file and environment variables."""
    try:
        print("Loading settings...")
        # Load config file
        config_path = Path("config.yaml")
        if not config_path.exists():
            print(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("Config file loaded")

        # Override settings from environment variables
        if "GEMINI_API_KEY" in os.environ:
            print("Found GEMINI_API_KEY in environment")
            config["api"]["gemini_api_key"] = os.environ["GEMINI_API_KEY"]
            
        if "LANGTRACE_API_KEY" in os.environ:
            print("Found LANGTRACE_API_KEY in environment")
            if "langtrace" not in config:
                config["langtrace"] = {}
            config["langtrace"]["api_key"] = os.environ["LANGTRACE_API_KEY"]
            config["langtrace"]["enabled"] = True
            print(f"LangTrace API key set: {config['langtrace']['api_key'][:8]}...")
            
        if "NOTION_API_KEY" in os.environ:
            print("Found NOTION_API_KEY in environment")
            if "notion" not in config:
                config["notion"] = {}
            config["notion"]["api_key"] = os.environ["NOTION_API_KEY"]
            print("Notion API key set")
            
        if "NOTION_DATABASE_ID" in os.environ:
            print("Found NOTION_DATABASE_ID in environment")
            if "notion" not in config:
                config["notion"] = {}
            config["notion"]["database_id"] = os.environ["NOTION_DATABASE_ID"]
            print(f"Notion database ID set: {os.environ['NOTION_DATABASE_ID']}")
            
        if "FREEIMAGE_API_KEY" in os.environ:
            print("Found FREEIMAGE_API_KEY in environment")
            if "freeimage" not in config:
                config["freeimage"] = {}
            config["freeimage"]["api_key"] = os.environ["FREEIMAGE_API_KEY"]
            print(f"Freeimage API key set: {os.environ['FREEIMAGE_API_KEY'][:8]}...")

        # Create settings instance
        settings = Settings(**config)
        print("Settings instance created")
        
        # Print all settings sections for debugging
        print("Available settings sections:", list(config.keys()))
        if "freeimage" in config:
            print("Freeimage settings:", config["freeimage"])
        
        return settings
        
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        logger.error(f"Error loading settings: {str(e)}", exc_info=True)
        raise

@lru_cache()
def get_settings() -> Settings:
    """Get the settings instance, loading it if necessary."""
    global _settings_instance
    if _settings_instance is None:
        print("Loading settings for the first time")
        _settings_instance = load_settings()
    return _settings_instance
