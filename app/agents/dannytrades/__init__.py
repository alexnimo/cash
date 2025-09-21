"""
DannyTrades Agent Module

This module provides the DannyTrades agent implementation for processing
stock market data and updating Notion databases via MCP tools.
"""

from .agent import DannyTradesAgent
from .config import DannyTradesConfig
from .models import (
    StockData,
    ProcessingResult,
    AgentActivity
)
from .utils import (
    PerformanceMonitor,
    SimpleCache,
    DataValidator,
    RateLimiter,
    performance_monitor,
    cache,
    rate_limiter
)
from .cli_agent import CLIAgent
from .cli_config import CLIConfig, AgentConfig

__all__ = [
    'DannyTradesAgent',
    'DannyTradesConfig',
    'StockData',
    'ProcessingResult',
    'AgentActivity',
    'PerformanceMonitor',
    'SimpleCache',
    'DataValidator',
    'RateLimiter',
    'performance_monitor',
    'cache',
    'rate_limiter',
    'CLIAgent',
    'CLIConfig',
    'AgentConfig'
]

__version__ = "1.0.0"
