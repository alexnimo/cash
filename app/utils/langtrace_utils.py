import functools
import logging
import time
import os
from typing import Any, Callable, TypeVar, Optional
import asyncio
from langtrace_python_sdk import langtrace, with_langtrace_root_span
from app.core.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("LangTrace utils module loaded")

# Type variables for function types
F = TypeVar('F', bound=Callable[..., Any])

# Global langtrace instance
_langtrace_instance = None

def get_langtrace():
    """Get the LangTrace instance."""
    global _langtrace_instance
    return _langtrace_instance

def init_langtrace() -> bool:
    """Initialize LangTrace with settings from environment"""
    global _langtrace_instance
    try:
        settings = get_settings()
        logger.debug("Got settings for LangTrace initialization")
        
        if not settings.langtrace.enabled:
            logger.info("LangTrace is disabled")
            _langtrace_instance = None
            return False

        # Initialize LangTrace with the API key and endpoint
        langtrace.init(api_key=settings.langtrace.api_key)
        _langtrace_instance = langtrace
        
        logger.info("LangTrace initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangTrace: {e}")
        _langtrace_instance = None
        return False

def setup_gemini():
    """Set up Gemini integration with LangTrace."""
    try:
        logger.debug("Setting up Gemini integration...")
        # Currently no specific setup needed for Gemini
        logger.info("Gemini integration set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up Gemini integration: {str(e)}", exc_info=True)
        return False

def trace_llm_call(operation_name: str) -> Callable[[F], F]:
    """
    Decorator to trace LLM API calls and log performance metrics.
    Works with both synchronous and asynchronous functions.
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            @with_langtrace_root_span(operation_name)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"LLM call {operation_name} completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    logger.error(f"Error in LLM API call {operation_name}: {str(e)}", exc_info=True)
                    raise
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            @with_langtrace_root_span(operation_name)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"LLM call {operation_name} completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    logger.error(f"Error in LLM API call {operation_name}: {str(e)}", exc_info=True)
                    raise
            return sync_wrapper  # type: ignore
            
    return decorator

def trace_gemini_call(operation_name: str) -> Callable[[F], F]:
    """Alias for trace_llm_call for Gemini-specific operations"""
    return trace_llm_call(operation_name)

# Initialize LangTrace at module level
init_success = init_langtrace()
if not init_success:
    logger.warning("LangTrace initialization failed. Tracing may not work correctly.")
