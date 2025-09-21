"""Configuration module for the application."""
from app.core.settings import get_settings, Settings
from pydantic import BaseModel

class TranscriptionConfig(BaseModel):
    enabled: bool = True
    name: str = "gemini-pro"
    max_output_tokens: int = 30000  # Default for 2 hours of audio (assuming ~250 words per minute)

__all__ = ['get_settings', 'Settings', 'TranscriptionConfig']