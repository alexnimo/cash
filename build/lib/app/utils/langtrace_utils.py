import functools
import logging
import time
from typing import Any, Callable, TypeVar, Optional
import asyncio
from app.core.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("LangTrace utils module loaded (stub implementation - tracing disabled)")

# Type variables for function types
F = TypeVar('F', bound=Callable[..., Any])

# No langtrace instance - always None
_langtrace_instance = None

def get_langtrace():
    """Get the LangTrace instance (always returns None)."""
    return None

def init_langtrace() -> bool:
    """Initialize LangTrace with settings from environment (stub implementation)"""
    logger.info("LangTrace is disabled (stub implementation)")
    return False

def setup_gemini():
    """Set up Gemini integration with LangTrace (stub implementation)."""
    logger.info("Gemini integration set up successfully (no tracing)")
    return True

# Helper function to create a no-op decorator
def no_op_decorator(func):
    """A decorator that does nothing but return the original function."""
    return func

# Stub implementation that mimics the with_langtrace_root_span decorator
def with_langtrace_root_span_stub(operation_name):
    """Stub implementation of with_langtrace_root_span"""
    return no_op_decorator

def trace_llm_call(operation_name: str) -> Callable[[F], F]:
    """
    Decorator to trace LLM API calls and log performance metrics.
    Works with both synchronous and asynchronous functions.
    LangTrace functionality has been disabled - this only provides timing logs now.
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
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

# No LangTrace initialization needed
logger.info("LangTrace initialization skipped - using stub implementation")
