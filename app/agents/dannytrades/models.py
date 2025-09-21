"""
Data models for DannyTrades agent.

This module defines the Pydantic models for stock data processing,
following the JSON schema specified in the PRD.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class BullishnessSignal(str, Enum):
    """Enumeration for bullishness signals."""
    EXTREME_BULLISH = "Extreme Bullish"
    HOT = "Hot"


class CandleTimeframe(BaseModel):
    """Model for candle timeframe data."""
    daily: bool = False
    weekly: bool = False
    monthly: bool = False


class WhaleAccumulation(BaseModel):
    """Model for whale accumulation data."""
    daily: str = ""
    weekly: str = ""
    monthly: str = ""


class TechnicalIndicators(BaseModel):
    """Model for technical indicators."""
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)


class StockData(BaseModel):
    """
    Model for individual stock data following the JSON schema.
    
    This model represents a single stock entry with all its associated
    market data and technical indicators.
    """
    symbol: str = Field(..., description="Stock symbol (required)")
    ticker_name: Optional[str] = Field(None, description="Full ticker name")
    volatility_hole: Optional[str] = Field(None, description="Volatility hole indicator")
    red_candle: Optional[CandleTimeframe] = Field(None, description="Red candle timeframes")
    yellow_candle: Optional[CandleTimeframe] = Field(None, description="Yellow candle timeframes")
    whale_accumulation: Optional[WhaleAccumulation] = Field(None, description="Whale accumulation data")
    technical_indicators: Optional[TechnicalIndicators] = Field(None, description="Technical analysis data")
    misc_data: Optional[str] = Field(None, description="Additional miscellaneous data")
    patreon_link: Optional[str] = Field(None, description="Patreon or reference link")
    bullishness_signal: List[BullishnessSignal] = Field(default_factory=list, description="Bullishness indicators")


class StockEmailExtraction(BaseModel):
    """
    Root model for stock email extraction data.
    
    This is the main container for all stock data received via the API endpoint.
    """
    stocks: List[StockData] = Field(..., description="List of stock data entries")


class ProcessingStatus(str, Enum):
    """Enumeration for processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingResult(BaseModel):
    """
    Model for agent processing results.
    
    This model tracks the results of processing stock data through the agent.
    """
    success: bool = Field(..., description="Whether processing was successful")
    action: Optional[str] = Field(None, description="Action taken by the agent")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for the action")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    stock_symbol: Optional[str] = Field(None, description="Stock symbol processed")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class AgentActivity(BaseModel):
    """
    Model for real-time agent activity tracking.
    
    This model is used for the real-time monitoring system.
    """
    activity_id: str = Field(..., description="Unique activity identifier")
    activity_type: str = Field(..., description="Type of activity (processing, updating, etc.)")
    stock_symbol: Optional[str] = Field(None, description="Stock symbol being processed")
    message: str = Field(..., description="Activity message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="info", description="Activity status (info, warning, error, success)")


class LLMProviderConfig(BaseModel):
    """
    Model for LLM provider configuration.
    
    This model defines the configuration structure for different LLM providers.
    """
    provider: str = Field(..., description="Provider name (ollama, claude, gemini, etc.)")
    model_id: str = Field(..., description="Model identifier")
    api_key: Optional[str] = Field(None, description="API key (if required)")
    endpoint: Optional[str] = Field(None, description="Custom endpoint URL")
    temperature: float = Field(default=0.2, description="Model temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens")
    enabled: bool = Field(default=True, description="Whether this provider is enabled")


class MCPServerConfig(BaseModel):
    """
    Model for MCP server configuration.
    
    This model defines the configuration for MCP servers used by the agent.
    """
    server_name: str = Field(..., description="MCP server name")
    command: str = Field(..., description="Command to start the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    enabled: bool = Field(default=True, description="Whether this server is enabled")


class AgentPromptConfig(BaseModel):
    """
    Model for agent prompt configuration.
    
    This model stores the customizable prompts used by the agent.
    """
    system_prompt: str = Field(..., description="Main system prompt for the agent")
    processing_instructions: str = Field(..., description="Specific processing instructions")
    error_handling_prompt: str = Field(..., description="Error handling instructions")
    updated_at: datetime = Field(default_factory=datetime.utcnow)
