from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import yaml
import os

router = APIRouter()

# Get the absolute path to config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")

class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    gemini_rpm_override: Optional[int] = None

@router.get("/settings")
async def get_settings():
    """Get current settings (excluding sensitive data)"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        
        return {
            "gemini_rpm_override": config.get("api", {}).get("gemini_rpm_override"),
            # Don't return API keys
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings")
async def update_settings(settings: SettingsUpdate):
    """Update settings"""
    try:
        # Load current config
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        
        # Update API settings
        if "api" not in config:
            config["api"] = {}
            
        if settings.gemini_api_key:
            # Update environment variable
            os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
            
        if settings.gemini_rpm_override is not None:
            if not 1 <= settings.gemini_rpm_override <= 60:
                raise HTTPException(status_code=400, detail="RPM must be between 1 and 60")
            config["api"]["gemini_rpm_override"] = settings.gemini_rpm_override
        
        # Save config
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
