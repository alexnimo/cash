"""LLM Factory module for creating LLM instances"""
from typing import Dict, Any, Optional
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai_like import OpenAILike
import os
import logging
import agentops

agentops_api_key = os.getenv("AGENTOPS_API_KEY")


logger = logging.getLogger(__name__)
agentops.init(agentops_api_key)

class LLMFactory:
    """Factory class for creating LLM instances"""
    
    @staticmethod
    def create_llm(config: Dict[str, Any]) -> Any:
        """Create an LLM instance based on configuration
        
        Args:
            config: Dictionary containing LLM configuration with:
                - provider: The LLM provider (openai, gemini, openai_like)
                - model: The model name
                - api_base: Optional API base URL for OpenAILike
                - api_key: Optional API key (if not set in environment)
                - Additional provider-specific parameters
        """
        provider = config.get('provider', 'openai').lower()
        model = config.get('model')
        api_key = config.get('api_key')
        
        try:
            if provider == 'openai':
                if api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                return OpenAI(
                    model=model or "gpt-4-turbo",
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens'),
                    additional_kwargs=config.get('additional_kwargs', {})
                )
                
            elif provider == 'gemini':
                if api_key:
                    os.environ['GOOGLE_API_KEY'] = api_key
                return Gemini(
                    model=model or "gemini-pro",
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens'),
                    additional_kwargs=config.get('additional_kwargs', {})
                )
                
            elif provider == 'openai_like':
                return OpenAILike(
                    model=model,
                    api_base=config.get('api_base'),
                    api_key=api_key,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens'),
                    additional_kwargs=config.get('additional_kwargs', {})
                )
                
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error creating LLM instance for provider {provider}: {str(e)}")
            raise
