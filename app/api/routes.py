"""API routes for the video analyzer application."""

from fastapi import APIRouter, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from app.services.model_manager import ModelManager
from app.core.config import get_settings
from typing import Optional, Dict, Any
from pydantic import BaseModel
import yaml
import os
from pathlib import Path

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
        
        return {
            "status": "success",
            "message": "LangTrace configuration updated successfully"
        }
    except Exception as e:
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
        return quota_info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )
