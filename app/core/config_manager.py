"""
Configuration manager for persistent settings.

Handles saving and loading user configurations that override defaults.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages persistent configuration settings."""
    
    def __init__(self, config_dir: str = ".cash_config"):
        """Initialize config manager with config directory."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.dannytrades_config_file = self.config_dir / "dannytrades_config.json"
        
    def save_dannytrades_config(self, config: Dict[str, Any]) -> bool:
        """
        Save DannyTrades configuration to disk.
        
        Args:
            config: Configuration dictionary to save
            
        Returns:
            True if saved successfully
        """
        try:
            with open(self.dannytrades_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved DannyTrades config to {self.dannytrades_config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save DannyTrades config: {e}")
            return False
    
    def load_dannytrades_config(self) -> Dict[str, Any]:
        """
        Load DannyTrades configuration from disk.
        
        Returns:
            Configuration dictionary or empty dict if not found
        """
        try:
            if self.dannytrades_config_file.exists():
                with open(self.dannytrades_config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Clean any potential control characters that cause JSON parsing issues
                    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                    config = json.loads(content)
                logger.info(f"Loaded DannyTrades config from {self.dannytrades_config_file}")
                return config
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in DannyTrades config: {e}")
            logger.error(f"Please check the JSON syntax in {self.dannytrades_config_file}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load DannyTrades config: {e}")
            return {}
    
    def update_provider_config(self, provider: str, **provider_settings: Any) -> bool:
        """Update provider configuration and save to disk."""
        try:
            config = self.load_dannytrades_config()

            if 'providers' not in config:
                config['providers'] = {}

            if provider not in config['providers']:
                config['providers'][provider] = {}

            # Update provider settings from kwargs, filtering out None values
            for key, value in provider_settings.items():
                if value is not None:
                    config['providers'][provider][key] = value

            config['providers'][provider]['enabled'] = True
            config['current_provider'] = provider

            return self.save_dannytrades_config(config)
        except Exception as e:
            logger.error(f"Failed to update provider config: {e}")
            return False
    
    def update_mcp_config(self, servers: Dict[str, Any]) -> bool:
        """
        Update MCP server configuration.
        
        Args:
            servers: MCP server configuration dictionary
            
        Returns:
            True if updated successfully
        """
        config = self.load_dannytrades_config()
        config['mcp_servers'] = servers
        return self.save_dannytrades_config(config)
    
    def get_active_provider(self) -> str:
        """Get the currently active LLM provider."""
        config = self.load_dannytrades_config()
        return config.get('default_provider', 'gemini')
    
    def get_current_provider_config(self) -> Dict[str, Any]:
        """Get the current active provider configuration."""
        config = self.load_dannytrades_config()
        current_provider = config.get('current_provider')
        providers = config.get('providers', {})
        
        if current_provider and current_provider in providers:
            return {
                'provider': current_provider,
                'model': providers[current_provider].get('model'),
                'api_key': providers[current_provider].get('api_key'),
                'endpoint': providers[current_provider].get('endpoint')
            }
        
        # Return empty config if no user selection exists
        return {
            'provider': None,
            'model': None,
            'api_key': None,
            'endpoint': None
        }
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        config = self.load_dannytrades_config()
        providers = config.get('llm_providers', {})
        return providers.get(provider, {})
    
    def get_mcp_servers(self) -> Dict[str, Any]:
        """Get MCP server configuration."""
        config = self.load_dannytrades_config()
        return config.get('mcp_servers', {})

# Global config manager instance
config_manager = ConfigManager()
