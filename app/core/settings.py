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
    transcription: ModelSettings
    embedding: ModelSettings

class VectorStoreConfig(BaseModel):
    type: str
    config: Dict[str, Any]
    metadata_config: Dict[str, List[str]] = Field(
        default={"indexed": ["video_id", "timestamp", "content_type", "language"]}
    )

class StorageConfig(BaseModel):
    transcript_dir: str
    frames_dir: str
    summaries_dir: str
    temp_dir: str

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

class Settings(BaseModel):
    app_name: str
    model: ModelConfig
    vector_store: VectorStoreConfig
    storage: StorageConfig
    api: APIConfig
    langtrace: LangTraceSettings
    processing: ProcessingConfig
    logging: LoggingSettings
    video_storage: VideoStorageSettings

_settings_instance = None

def load_settings() -> Settings:
    """Load settings from config file and environment variables."""
    try:
        print("Loading settings...")
        # Load environment variables
        load_dotenv()
        print("Environment variables loaded")

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

        # Create settings instance
        settings = Settings(**config)
        print("Settings instance created")
        
        # Verify LangTrace settings
        if settings.langtrace.enabled:
            if not settings.langtrace.api_key:
                print("Warning: LangTrace is enabled but API key is not set")
            else:
                print(f"LangTrace is enabled with API key: {settings.langtrace.api_key[:8]}...")
        
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
