from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import google.generativeai as genai
from app.core.config import get_settings
from app.utils.langtrace_utils import get_langtrace, trace_llm_call
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self):
        logger.info("Initializing BaseAgent")
        self.settings = get_settings()
        self._setup_model()
        self._langtrace = get_langtrace()
        if self._langtrace:
            logger.info("LangTrace is available for agent tracing")
        else:
            logger.warning("LangTrace is not available, agent tracing will be disabled")
        
    def _setup_model(self):
        """Setup the LLM model according to configuration."""
        logger.info("Setting up LLM model")
        try:
            genai.configure(api_key=self.settings.api.gemini_api_key)
            logger.debug("Configured Gemini API key")
            
            model_config = self.settings.agents.llm
            generation_config = {
                "temperature": model_config.temperature
            }
            logger.debug(f"Using model configuration: {model_config.name}, temperature: {model_config.temperature}")
            
            self.model = genai.GenerativeModel(
                model_config.name,
                generation_config=generation_config
            )
            logger.info("LLM model setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to setup LLM model: {str(e)}", exc_info=True)
            raise
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        """Execute the agent's task."""
        pass
