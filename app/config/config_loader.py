"""
Configuration loader for the video analyzer application.
Provides access to settings from config.yaml with environment variable support.
"""
from typing import Dict, Any, Optional, List, Union
import os
import yaml
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads and provides access to application configuration from YAML
    with environment variable substitution.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.
        
        Returns:
            Dict containing the parsed configuration
        """
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                # Load the YAML content
                config = yaml.safe_load(f)
                
                # Process environment variables in the config
                config = self._process_env_vars(config)
                
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def _process_env_vars(self, config: Any) -> Any:
        """
        Process environment variables in configuration values.
        
        Replaces ${ENV_VAR} or $ENV_VAR with the actual environment variable value.
        
        Args:
            config: Configuration object (could be dict, list, or scalar value)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._process_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern to match ${VAR} or $VAR
            pattern = r'\${([A-Za-z0-9_]+)}|\$([A-Za-z0-9_]+)'
            
            def replace_env_var(match):
                var_name = match.group(1) or match.group(2)
                return os.environ.get(var_name, match.group(0))
                
            return re.sub(pattern, replace_env_var, config)
        else:
            return config
            
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.
        
        Args:
            *keys: Key path components (e.g., 'agents', 'rag', 'llm')
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
        """
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current

# Create a singleton instance
_config_loader = None

def get_config_loader(config_path: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """
    Get the singleton ConfigLoader instance.
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        if config_path is None:
            # Default to project root config.yaml
            config_path = Path(__file__).parents[2] / 'config.yaml'
        
        _config_loader = ConfigLoader(config_path)
        
    return _config_loader
