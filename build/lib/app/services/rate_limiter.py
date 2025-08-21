"""Rate limiter implementation for API requests."""

import time
from collections import deque
import logging
from typing import Deque, Tuple
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 15, time_window: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests: Deque[float] = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Tuple[bool, float]:
        """Try to acquire a rate limit slot.
        
        Returns:
            Tuple of (can_proceed, wait_time)
            - can_proceed: Whether the request can proceed
            - wait_time: Time to wait in seconds before next request
        """
        async with self._lock:
            now = time.time()
            
            # Remove requests older than the time window
            while self.requests and now - self.requests[0] >= self.time_window:
                self.requests.popleft()
            
            # Check if we've hit the rate limit
            if len(self.requests) >= self.max_requests:
                # Calculate wait time until oldest request expires
                wait_time = self.time_window - (now - self.requests[0])
                logger.warning(f"Rate limit reached. Need to wait {wait_time:.2f} seconds")
                return False, wait_time
            
            # Add current request
            self.requests.append(now)
            return True, 0
    
    def get_status(self) -> dict:
        """Get current rate limit status."""
        now = time.time()
        
        # Clean up old requests
        while self.requests and now - self.requests[0] >= self.time_window:
            self.requests.popleft()
        
        requests_in_window = len(self.requests)
        remaining_requests = self.max_requests - requests_in_window
        
        # Calculate reset time
        if requests_in_window > 0:
            time_until_reset = self.time_window - (now - self.requests[0])
        else:
            time_until_reset = 0
            
        return {
            "total_limit": self.max_requests,
            "remaining": remaining_requests,
            "requests_in_window": requests_in_window,
            "reset_in_seconds": round(time_until_reset),
            "window_size": self.time_window,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
