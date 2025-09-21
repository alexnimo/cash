"""Model configuration module for initializing AI models."""

import os
import time
import google.generativeai as genai
from google.api_core import retry
from app.core.config import get_settings
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)
settings = get_settings()

# Constants for retry configuration
MAX_RETRIES = 5
RETRY_INTERVAL = 60  # 1 minute between retries
SERVICE_ENDPOINTS = [
    "generativelanguage.googleapis.com"
]

def _is_retryable_error(exception) -> bool:
    """Check if an error should trigger a retry."""
    if hasattr(exception, 'details'):
        error_details = str(exception.details).lower()
        return any(condition in error_details for condition in [
            'rate limit',
            'resource exhausted',
            'deadline exceeded',
            'unavailable',
            'service unavailable',
            '503',
            'temporarily unavailable'
        ])
    return False

def _check_service_availability(endpoint: str) -> bool:
    """Check if a service endpoint is available."""
    try:
        response = requests.get(f"https://{endpoint}", timeout=5)
        return response.status_code != 503
    except:
        return False

def configure_models(max_retries: int = MAX_RETRIES, retry_interval: float = RETRY_INTERVAL):
    """Configure all AI models with appropriate settings and retry logic."""
    try:
        logger.info("Configuring Gemini model")
        
        # Get API key from environment first, then settings
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            # Try to get from settings, but handle the template case
            api_key = settings.api.gemini_api_key
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]  # Remove ${ and }
                api_key = os.getenv(env_var)
        
        if not api_key:
            raise ValueError("Gemini API key not found in environment variables or settings")
        
        # Try each service endpoint
        last_error = None
        for endpoint in SERVICE_ENDPOINTS:
            logger.info(f"Attempting to configure with endpoint: {endpoint}")
            
            # Configure with retries
            for attempt in range(max_retries):
                try:
                    # Check service availability
                    if not _check_service_availability(endpoint):
                        logger.warning(f"Service endpoint {endpoint} appears to be unavailable")
                        raise Exception("Service temporarily unavailable")
                    
                    # Configure Gemini without making any API calls
                    genai.configure(
                        api_key=api_key,
                        transport="rest",
                        client_options={"api_endpoint": f"https://{endpoint}"}
                    )
                    
                    logger.info(f"Successfully configured Gemini model with endpoint {endpoint}")
                    return
                    
                except Exception as e:
                    last_error = e
                    if attempt == max_retries - 1:  # Last attempt for this endpoint
                        logger.warning(f"All retries failed for endpoint {endpoint}: {str(e)}")
                        continue
                    
                    if _is_retryable_error(e):
                        logger.warning(f"Retryable error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        logger.info(f"Waiting {retry_interval} seconds before retry...")
                        time.sleep(retry_interval)
                    else:
                        # Non-retryable error, try next endpoint
                        logger.warning(f"Non-retryable error for endpoint {endpoint}: {str(e)}")
                        break
        
        # If we get here, all endpoints failed
        raise last_error or Exception("Failed to configure Gemini with any available endpoint")
                    
    except Exception as e:
        logger.error(f"Failed to configure models: {str(e)}", exc_info=True)
        raise
