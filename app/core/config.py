"""Configuration module for the application."""
import yaml
from pathlib import Path
from typing import Dict, Any
from app.core.settings import get_settings, Settings
from pydantic import BaseModel

class TranscriptionConfig(BaseModel):
    enabled: bool = True
    name: str = "gemini-pro"
    max_output_tokens: int = 30000  # Default for 2 hours of audio (assuming ~250 words per minute)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}

__all__ = ['get_settings', 'Settings', 'TranscriptionConfig', 'load_config']