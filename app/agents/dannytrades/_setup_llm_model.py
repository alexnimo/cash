"""
LLM model setup utilities for DannyTrades agent.
"""

import logging
from typing import Any, Optional

from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.ollama import Ollama

logger = logging.getLogger(__name__)


async def setup_llm_model(config) -> tuple[Any, str]:
    """
    Setup the LLM model based on configuration.
    
    Args:
        config: DannyTradesConfig instance
        
    Returns:
        Tuple of (model_instance, provider_name)
        
    Raises:
        ValueError: If no valid LLM configuration is found
    """
    # Validate configuration first
    is_valid, error_message = config.validate_llm_config()
    if not is_valid:
        raise ValueError(f"LLM Configuration Error: {error_message}")
    
    provider = config.default_provider
    
    if provider == "openai":
        # Get provider config to check for custom endpoint
        provider_config = config.get_provider_config("openai")
        custom_endpoint = provider_config.get('endpoint')
        
        # Handle custom OpenAI-compatible endpoints
        if custom_endpoint and custom_endpoint != "https://api.openai.com/v1":
            # For custom endpoints, we might not need an API key or use a placeholder
            api_key = config.openai_api_key or "placeholder-key"
            model = OpenAIChat(
                id=config.default_model,
                api_key=api_key,
                base_url=custom_endpoint
            )
        else:
            # Standard OpenAI endpoint requires API key
            if not config.openai_api_key:
                raise ValueError("OpenAI API key is required for standard OpenAI endpoint")
            model = OpenAIChat(
                id=config.default_model,
                api_key=config.openai_api_key
            )
    elif provider == "anthropic":
        model = Claude(
            id=config.default_model,
            api_key=config.anthropic_api_key
        )
    elif provider == "ollama":
        # For Ollama, use the configured model or raise error if none specified
        if not config.default_model:
            raise ValueError("No model specified for Ollama provider. Please configure a model in config.yaml under dannytrades.llm.default_model")
        
        # Use custom Ollama client for tool calling support
        from .custom_ollama_client import CustomOllamaModel
        model = CustomOllamaModel(
            id=config.default_model,
            host="http://localhost:11434"
        )
    elif provider == "gemini":
        model = Gemini(
            id=config.default_model,
            api_key=config.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    logger.info(f"Initialized LLM model: {provider} with model: {config.default_model}")
    return model, provider
