"""API routes for the video analyzer application."""

from fastapi import APIRouter, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from app.services.model_manager import ModelManager
from app.core.config import get_settings
from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.agents.agents import AgentWorkflow, TechnicalAnalysisAgent
from app.config.agent_config import AGENT_CONFIG
from app.tools import notion_tool_v2
import httpx
from llama_index.core.memory import ChatMemoryBuffer
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

class LangTraceConfig(BaseModel):
    enabled: bool
    api_key: Optional[str] = None

def get_model_manager():
    """Get or create the model manager instance."""
    return ModelManager(settings)

@router.get("/api/langtrace/config")
async def get_langtrace_config() -> Dict[str, Any]:
    """Get the current LangTrace configuration."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return {
            "enabled": config.get("langtrace", {}).get("enabled", False),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in getting LangTrace config: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@router.post("/api/langtrace/config")
async def update_langtrace_config(config: LangTraceConfig) -> Dict[str, Any]:
    """Update LangTrace configuration."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    try:
        # Read existing config
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
        
        # Update LangTrace settings
        if "langtrace" not in full_config:
            full_config["langtrace"] = {}
        
        full_config["langtrace"]["enabled"] = config.enabled
        
        if config.api_key:
            # Update .env file with new API key
            env_path = config_path.parent / ".env"
            env_lines = []
            api_key_updated = False
            
            if env_path.exists():
                with open(env_path, "r") as f:
                    env_lines = f.readlines()
                    
                # Update existing LANGTRACE_API_KEY if found
                for i, line in enumerate(env_lines):
                    if line.startswith("LANGTRACE_API_KEY="):
                        env_lines[i] = f"LANGTRACE_API_KEY={config.api_key}\n"
                        api_key_updated = True
                        break
            
            # Add new LANGTRACE_API_KEY if not found
            if not api_key_updated:
                env_lines.append(f"\n# LangTrace settings\nLANGTRACE_API_KEY={config.api_key}\n")
            
            # Write updated .env file
            with open(env_path, "w") as f:
                f.writelines(env_lines)
            
            # Update config.yaml to reference env variable
            full_config["langtrace"]["api_key"] = "${LANGTRACE_API_KEY}"
        
        # Write updated config
        with open(config_path, "w") as f:
            yaml.safe_dump(full_config, f, default_flow_style=False)
        
        logger.info("LangTrace configuration updated successfully")
        return {
            "status": "success",
            "message": "LangTrace configuration updated successfully"
        }
    except Exception as e:
        logger.error(f"Error in updating LangTrace config: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@router.get("/api/quota")
async def check_api_quota():
    """Check the remaining quota for the Gemini API key."""
    try:
        model_manager = get_model_manager()
        quota_info = await model_manager.check_quota()
        logger.info("API quota checked successfully")
        return quota_info
    except Exception as e:
        logger.error(f"Error in checking API quota: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

agent_router = APIRouter()

# Initialize agents lazily
_agent_workflow = None
_ta_agent = None

async def get_agent_workflow():
    global _agent_workflow
    if _agent_workflow is None:
        _agent_workflow = AgentWorkflow(AGENT_CONFIG)
    return _agent_workflow

async def get_ta_agent():
    global _ta_agent
    if _ta_agent is None:
        # Initialize required tools
        notion_tool = notion_tool_v2.NotionTool()
        tools = [notion_tool]
        
        _ta_agent = TechnicalAnalysisAgent(
            tools=tools,
            llm=ModelManager.create_llm(AGENT_CONFIG['technical_analysis']['llm']),
            memory=ChatMemoryBuffer.from_defaults(),
            config=AGENT_CONFIG['technical_analysis']
        )
    return _ta_agent

@agent_router.post("/process-video-summary")
async def process_video_summary(summary_file: str):
    """Process a video summary file after video processing is complete"""
    try:
        agent = await get_agent_workflow()
        result = await agent.process_summary(summary_file)
        logger.info("Video summary processed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in processing video summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/test-technical-analysis")
async def test_technical_analysis(analysis_data: Dict = Body(...)):
    """Test endpoint for technical analysis agent"""
    try:
        logger.info(f"Received analysis data: {analysis_data}")
        agent = await get_ta_agent()
        logger.info("Agent initialized successfully")
        result = await agent.execute(analysis_data)
        logger.info(f"Analysis completed successfully: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

@agent_router.post("/test-technical-analysis-from-url")
async def process_technical_analysis_from_url(url: str):
    """Process technical analysis from JSON URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            analysis_data = response.json()
        
        agent = await get_ta_agent()
        result = await agent.execute(analysis_data)
        logger.info("Technical analysis from URL processed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in processing technical analysis from URL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

router.include_router(agent_router, prefix="/api/agents")
