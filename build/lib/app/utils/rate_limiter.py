import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int, period: timedelta, callback: Optional[Callable] = None):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the period
            period: Time period for the rate limit (as timedelta)
            callback: Optional callback function to call when rate limit is hit
        """
        self.max_requests = max_requests
        self.period = period
        self.callback = callback
        self.request_times = []
        self.interval = period.total_seconds() / max_requests
        
        logger.info(f"Initialized RateLimiter with {max_requests} requests per {period} "
                   f"(interval: {self.interval:.2f}s)")

    def allow_request(self) -> bool:
        """Check if a request is allowed at the current time."""
        now = datetime.now()
        window_start = now - self.period
        
        # Remove old requests
        self.request_times = [t for t in self.request_times if t > window_start]
        
        # Check if we're under the limit
        if len(self.request_times) < self.max_requests:
            self.request_times.append(now)
            return True
            
        # We're at the limit, calculate wait time
        if self.callback:
            self.callback()
        return False

    async def wait(self) -> None:
        """Wait until a request is allowed."""
        while not self.allow_request():
            # Calculate time until next request slot
            oldest_request = self.request_times[0]
            wait_time = (oldest_request + self.period - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.request_times = self.request_times[1:]  # Remove the oldest request
