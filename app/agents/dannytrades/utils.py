"""
Utility functions for DannyTrades Agent.

This module contains helper functions for data processing, validation,
caching, and performance optimization.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics for the agent."""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'response_times': [],
            'last_reset': datetime.utcnow()
        }
    
    def record_request(self, success: bool, response_time: float):
        """Record a request with its outcome and response time."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        self.metrics['response_times'].append(response_time)
        
        # Keep only last 100 response times for memory efficiency
        if len(self.metrics['response_times']) > 100:
            self.metrics['response_times'] = self.metrics['response_times'][-100:]
        
        # Update average response time
        self.metrics['average_response_time'] = sum(self.metrics['response_times']) / len(self.metrics['response_times'])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        success_rate = 0.0
        if self.metrics['total_requests'] > 0:
            success_rate = (self.metrics['successful_requests'] / self.metrics['total_requests']) * 100
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'error_rate': 100 - success_rate
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_metrics() for backward compatibility."""
        return self.get_metrics()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'response_times': [],
            'last_reset': datetime.utcnow()
        }


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, data: Union[str, Dict[str, Any]]) -> str:
        """Generate a cache key from data."""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        if cache_key in self.cache:
            value, expiry = self.cache[cache_key]
            if datetime.utcnow() < expiry:
                return value
            else:
                del self.cache[cache_key]
        
        return None
    
    def set(self, key: Union[str, Dict[str, Any]], value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        ttl = ttl or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self.cache[cache_key] = (value, expiry)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries from cache."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self.cache[key]


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for async functions to add retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {current_delay}s: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class DataValidator:
    """Utility class for data validation and sanitization."""
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize stock symbol."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Remove special characters and convert to uppercase
        sanitized = ''.join(c for c in symbol.upper() if c.isalnum())
        
        if not sanitized:
            raise ValueError("Symbol contains no valid characters")
        
        if len(sanitized) > 10:
            raise ValueError("Symbol too long (max 10 characters)")
        
        return sanitized
    
    @staticmethod
    def validate_price_levels(levels: List[float]) -> List[float]:
        """Validate and sanitize price levels."""
        if not isinstance(levels, list):
            raise ValueError("Price levels must be a list")
        
        if not levels:
            raise ValueError("Price levels cannot be empty")
        
        validated_levels = []
        for level in levels:
            if not isinstance(level, (int, float)):
                raise ValueError("Price levels must be numeric")
            
            if level <= 0:
                raise ValueError("Price levels must be positive")
            
            validated_levels.append(float(level))
        
        return sorted(validated_levels)
    
    @staticmethod
    def validate_confidence_score(confidence: float) -> float:
        """Validate confidence score."""
        if not isinstance(confidence, (int, float)):
            raise ValueError("Confidence must be numeric")
        
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        return float(confidence)
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitize text input."""
        if not isinstance(text, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        
        return sanitized.strip()


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self) -> bool:
        """
        Acquire permission to make a call.
        
        Returns:
            True if call is allowed, False if rate limited
        """
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Check if we can make another call
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        if not self.calls:
            return 0.0
        
        oldest_call = min(self.calls)
        return max(0.0, self.time_window - (time.time() - oldest_call))


def batch_process(items: List[Any], batch_size: int = 10):
    """
    Generator to process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


async def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON text with error handling.
    
    Args:
        text: JSON text to parse
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        
        # Try to extract JSON from text (common with LLM responses)
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {remaining_minutes:.0f}m"


def create_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error: Exception that occurred
        context: Additional context about the error
        
    Returns:
        Standardized error response dictionary
    """
    return {
        "success": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# Global instances for shared use
performance_monitor = PerformanceMonitor()
cache = SimpleCache()
rate_limiter = RateLimiter(max_calls=100, time_window=60)  # 100 calls per minute
