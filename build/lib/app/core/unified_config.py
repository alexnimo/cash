"""
Unified Configuration System for Video Analyzer Application.

This module provides a single, consistent way to access all application settings,
replacing the fragmented approach using multiple configuration files.

Features:
- Configuration from YAML files
- Environment variable substitution
- Type validation
- Dot notation access
- Backward compatibility with existing settings.py
"""
from typing import Dict, Any, Optional, Union, get_type_hints
from pathlib import Path
import os
import yaml
import json
import logging
from pydantic import BaseModel, create_model, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")

class ConfigManager:
    """
    Unified configuration manager for the video analyzer application.
    Combines functionality from settings.py and config_loader.py into
    a single, consistent interface.
    """
    
    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Set default config path if not provided
        if config_path is None:
            self.config_path = Path(__file__).parents[2] / 'config.yaml'
        else:
            self.config_path = Path(config_path)
            
        # Load configuration
        self.config = self._load_config()
        
        # Create Pydantic models for validated access
        self.models = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable substitution.
        
        Returns:
            Dict containing the loaded configuration
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
                
                # Print the raw config for debugging
                logger.debug(f"Raw config loaded: {json.dumps(config, default=str, indent=2)}")
                
                # Force specific boolean conversions for common boolean fields
                if 'agents' in config and isinstance(config['agents'], dict):
                    if 'enabled' in config['agents']:
                        # Force conversion to boolean
                        is_enabled = config['agents']['enabled']
                        if isinstance(is_enabled, str):
                            config['agents']['enabled'] = is_enabled.lower() in ['true', 'yes', '1', 'on']
                        else:
                            config['agents']['enabled'] = bool(is_enabled)
                        logger.info(f"Agents enabled value (after conversion): {config['agents']['enabled']}")
                
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _process_env_vars(self, config: Any) -> Any:
        """
        Process environment variables in configuration.
        
        Replaces ${ENV_VAR} or $ENV_VAR with the actual environment variable value.
        
        Args:
            config: Configuration object (dict, list, or scalar value)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._process_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Process ${VAR} style references
            if config.startswith('${') and config.endswith('}'):
                env_var = config[2:-1]
                return os.environ.get(env_var, config)
            
            # Process any embedded ${VAR} references
            import re
            pattern = r'\${([A-Za-z0-9_]+)}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
                
            return re.sub(pattern, replace_env_var, config)
        else:
            return config
    
    def reload(self):
        """Reload configuration from the config file."""
        self.config = self._load_config()
        self.models = {}  # Clear cached models
        logger.info(f"Reloaded configuration from {self.config_path}")
        return self.config
    
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
        
        # Special case for boolean values to ensure proper handling
        if isinstance(current, str) and keys and keys[-1] in ['enabled', 'disabled']:
            # Convert string representations of booleans
            if current.lower() in ['true', 'yes', '1', 'on']:
                return True
            elif current.lower() in ['false', 'no', '0', 'off']:
                return False
                
        return current
    
    def set(self, value: Any, *keys: str) -> None:
        """
        Set a configuration value by dot-notation path.
        
        Args:
            value: Value to set
            *keys: Key path components (e.g., 'agents', 'rag', 'llm')
        """
        if not keys:
            raise ValueError("No keys provided")
            
        current = self.config
        
        # Navigate to the nested dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
                
        # Set the value
        current[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save to (defaults to original config path)
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def get_model(self, *section_path: str) -> BaseModel:
        """
        Get a Pydantic model for a configuration section.
        Models are created dynamically based on the structure of the configuration.
        
        Args:
            *section_path: Path to the configuration section
            
        Returns:
            Pydantic BaseModel for the section
        """
        # Create a cache key from the section path
        cache_key = '.'.join(section_path)
        
        # Return cached model if available
        if cache_key in self.models:
            return self.models[cache_key]
            
        # Get the configuration section
        section = self.get(*section_path)
        if section is None:
            raise ValueError(f"Configuration section not found: {cache_key}")
            
        if not isinstance(section, dict):
            raise ValueError(f"Cannot create model for non-dict section: {cache_key}")
            
        # Create field definitions dynamically
        fields = {}
        for key, value in section.items():
            if isinstance(value, dict):
                # Nested model
                nested_path = list(section_path) + [key]
                field_type = self.get_model(*nested_path)
            else:
                # Regular field
                field_type = type(value)
                
            fields[key] = (field_type, Field(default=value))
            
        # Create the model dynamically
        model_name = 'Model' + ''.join(p.capitalize() for p in section_path)
        model = create_model(model_name, **fields)
        
        # Cache the model
        self.models[cache_key] = model
        
        # Instantiate the model with the current configuration values
        return model(**section)
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to top-level configuration sections.
        
        Args:
            name: Name of the top-level section
            
        Returns:
            Configuration section or dynamic model if section exists
        """
        if name in self.config:
            # If it's a dict section, return a model for type validation
            if isinstance(self.config[name], dict):
                try:
                    return self.get_model(name)
                except Exception:
                    return self.config[name]
            return self.config[name]
        
        raise AttributeError(f"Configuration has no section '{name}'")

# Singleton instance
_config_manager = None

def get_config() -> ConfigManager:
    """
    Get the singleton ConfigManager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
        
    return _config_manager

def reload_config() -> None:
    """
    Reload the configuration from the config file.
    Also updates the settings.py settings for backward compatibility.
    """
    config = get_config()
    config.reload()
    
    # Update settings.py for backward compatibility
    try:
        from app.core.settings import reload_settings
        reload_settings()
    except Exception as e:
        logger.warning(f"Failed to update settings.py: {str(e)}")
    
    return config
