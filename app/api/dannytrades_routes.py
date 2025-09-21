"""
API routes for DannyTrades agent functionality.

This module defines the FastAPI routes for processing stock data
through the DannyTrades agent system.
"""

import asyncio
import logging
import time
import uuid
import traceback
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.agents.dannytrades.simple_agent import SimpleDannyTradesAgent
from app.agents.dannytrades.models import StockData, ProcessingResult, StockEmailExtraction
from app.agents.dannytrades.utils import performance_monitor, timing_decorator, create_error_response
from app.agents.dannytrades.config import DannyTradesConfig
from app.agents.dannytrades.cli_agent import CLIAgent
from app.agents.dannytrades.cli_config import CLIConfig

# Create router for DannyTrades endpoints
router = APIRouter(prefix="/api/dannytrades", tags=["dannytrades"])

# Initialize logger
logger = logging.getLogger(__name__)

# Global agent instance (will be initialized on startup)
agent_instance: SimpleDannyTradesAgent = None

# CLI agent instance
cli_agent_instance: CLIAgent = None

# Processing results storage (in production, use Redis or database)
processing_results: Dict[str, ProcessingResult] = {}


async def get_dannytrades_agent() -> SimpleDannyTradesAgent:
    """
    Dependency function to get the initialized DannyTrades agent instance.
    """
    global agent_instance
    if agent_instance is None:
        try:
            # The SimpleDannyTradesAgent initializes itself on creation
            agent_instance = SimpleDannyTradesAgent()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize Simple DannyTrades agent: {str(e)}"
            )
    return agent_instance


async def get_cli_agent() -> CLIAgent:
    """
    Dependency function to get the initialized CLI agent instance.
    """
    global cli_agent_instance
    if cli_agent_instance is None:
        try:
            cli_agent_instance = CLIAgent()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize CLI agent: {str(e)}"
            )
    return cli_agent_instance




@router.post("/process")
@timing_decorator
async def process_stock_data(
    stock_data: Dict[str, Any],
    agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)
):
    """
    Process stock data through the DannyTrades agent.
    
    Args:
        stock_data: Stock data dictionary
        agent: DannyTrades agent instance
        
    Returns:
        Processing result
    """
    start_time = time.time()
    
    try:
        # Process the stock data
        result = await agent.process_stock_data(stock_data)
        
        # Record performance metrics
        processing_time = time.time() - start_time
        performance_monitor.record_request(result.success, processing_time)
        
        if result.success:
            return {
                "success": True,
                "action": result.action,
                "reasoning": result.reasoning,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "stock_symbol": result.stock_symbol
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error_message": result.error_message,
                    "processing_time": result.processing_time,
                    "stock_symbol": result.stock_symbol
                }
            )
        
    except ValidationError as e:
        performance_monitor.record_request(False, time.time() - start_time)
        raise HTTPException(status_code=422, detail=f"Invalid stock data: {str(e)}")
    except Exception as e:
        performance_monitor.record_request(False, time.time() - start_time)
        raise HTTPException(status_code=500, detail=create_error_response(e, "Stock processing"))


@router.get("/status")
async def get_dannytrades_agent_status(
    agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)
):
    """
    Get the current agent status and performance metrics.
    
    Args:
        agent: DannyTrades agent instance
        
    Returns:
        Agent status and performance data
    """
    try:
        status = await agent.get_status()
        metrics = performance_monitor.get_stats()
        
        return {
            "agent_status": status,
            "performance_metrics": metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in get_dannytrades_agent_status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/activities")
async def get_dannytrades_agent_activities(
    limit: int = 50,
    agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)
):
    """
    Get recent agent activities for real-time monitoring.
    
    Args:
        limit: Maximum number of activities to return
        agent: DannyTrades agent instance
        
    Returns:
        List of recent agent activities
    """
    try:
        activities = await agent.get_activities(limit)
        return {"activities": activities, "count": len(activities)}
    except Exception as e:
        logger.error(f"Error in get_dannytrades_agent_activities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get activities: {str(e)}")


@router.get("/providers/status")
async def get_provider_status(agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)):
    """
    Get status of all LLM providers.
    
    Args:
        agent: DannyTrades agent instance
        
    Returns:
        Provider status information
    """
    return agent.get_provider_status()


@router.post("/providers/switch")
async def switch_provider(
    provider_data: Dict[str, str],
    agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)
):
    """
    Switch the LLM provider for the DannyTrades agent.
    
    Args:
        provider_data: Dictionary containing provider configuration
        agent: DannyTrades agent instance
        
    Returns:
        JSON response with success status
    """
    provider_name = provider_data.get("provider")
    api_key = provider_data.get("api_key", "")
    model = provider_data.get("model", "")
    endpoint = provider_data.get("endpoint", "")
    user_agent = provider_data.get("user_agent", "")
    
    if not provider_name:
        raise HTTPException(status_code=400, detail="Provider name is required")

    try:
        # Pass the whole dictionary to the agent method
        success = await agent.switch_provider(**provider_data)
        if success:
            return {"status": "success", "message": f"Successfully switched to {provider_name} provider"}
        else:
            raise HTTPException(status_code=500, detail="Failed to switch provider")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching provider: {str(e)}")


@router.get("/mcp/status")
async def get_mcp_status(agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)):
    """
    Get status of all MCP servers.
    
    Args:
        agent: DannyTrades agent instance
        
    Returns:
        MCP server status information
    """
    return agent.get_mcp_server_status()


@router.post("/prompts/update")
async def update_prompts(
    prompt_data: Dict[str, str],
    agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)
):
    """
    Update agent prompts.
    
    Args:
        prompt_data: Dictionary with prompt fields
        agent: DannyTrades agent instance
        
    Returns:
        Success message
    """
    try:
        system_prompt = prompt_data.get("system_prompt", "")
        processing_instructions = prompt_data.get("processing_instructions", "")
        error_handling_prompt = prompt_data.get("error_handling_prompt", "")
        
        if not all([system_prompt, processing_instructions, error_handling_prompt]):
            raise HTTPException(status_code=400, detail="All prompt fields are required")
        
        agent.update_prompts(system_prompt, processing_instructions, error_handling_prompt)
        
        return {"status": "success", "message": "Prompts updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update prompts: {str(e)}")


@router.get("/config")
async def get_dannytrades_agent_config(agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)):
    """
    Get current agent configuration.
    
    Args:
        agent: DannyTrades agent instance
        
    Returns:
        Agent configuration information
    """
    try:
        provider_status = agent.get_provider_status()
        return {
            "current_provider": provider_status.get("current", "gemini"),
            "current_model": provider_status.get("model_name", "gemini-2.5-flash"),
            "available_providers": provider_status.get("available", ["gemini", "openai", "anthropic"]),
            "model_name": provider_status.get("model_name", "gemini-2.5-flash"),
            "status": provider_status.get("status", "inactive")
        }
    except Exception as e:
        logger.error(f"Error getting agent config: {e}")
        return {
            "current_provider": "gemini",
            "current_model": "gemini-2.5-flash",
            "available_providers": ["gemini", "openai", "anthropic"],
            "model_name": "gemini-2.5-flash",
            "status": "inactive",
            "error": str(e)
        }


@router.post("/test")
async def test_agent(
    test_data: Dict[str, Any],
    agent: SimpleDannyTradesAgent = Depends(get_dannytrades_agent)
):
    """
    Test endpoint for agent functionality.
    
    Args:
        test_data: Test data
        agent: DannyTrades agent instance
        
    Returns:
        Test result
    """
    try:
        symbol = test_data.get("symbol", "TEST")
        result = await agent.test_analysis(symbol)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")




@router.post("/agent-analyze")
@timing_decorator
async def agent_analyze(
    data: Dict[str, Any],
    cli_agent: CLIAgent = Depends(get_cli_agent)
):
    """
    Analyze data using CLI agents (Claude Code, Gemini CLI, etc.).

    Args:
        data: JSON data to analyze
        cli_agent: CLI agent instance

    Returns:
        Analysis result from CLI agent
    """
    start_time = time.time()

    try:
        # Process the data through CLI agent
        result = await cli_agent.analyze(data)

        # Record performance metrics
        processing_time = time.time() - start_time
        performance_monitor.record_request(result.get("success", False), processing_time)

        return result

    except Exception as e:
        performance_monitor.record_request(False, time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")


@router.post("/test-claude-cli")
async def test_claude_cli():
    """
    Test Claude CLI directly.

    Returns:
        Test result showing Claude CLI status and version
    """
    try:
        import subprocess
        import json
        import shutil

        # Find Claude CLI path
        claude_path = shutil.which("claude")
        if not claude_path:
            return {
                "success": False,
                "error": "Claude CLI not found in PATH",
                "hint": "Please install Claude CLI or ensure it's in your PATH"
            }

        # Test Claude version
        result = subprocess.run(
            [claude_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Test a simple prompt
        test_prompt = "Hello, please respond with 'Claude CLI is working'"
        prompt_result = subprocess.run(
            [claude_path, "-p", test_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        return {
            "success": True,
            "claude_path": claude_path,
            "version_test": {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            },
            "prompt_test": {
                "return_code": prompt_result.returncode,
                "stdout": prompt_result.stdout,
                "stderr": prompt_result.stderr
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "dannytrades-agent",
        "agent_initialized": agent_instance is not None,
        "cli_agent_initialized": cli_agent_instance is not None
    }
