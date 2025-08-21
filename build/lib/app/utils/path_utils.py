import os
import re
from pathlib import Path
from urllib.parse import unquote
from functools import lru_cache
from typing import Optional, Union, List

def normalize_path(path: str) -> str:
    """
    Normalize a file path to be compatible with the current operating system.
    Handles:
    - URL encoding/decoding
    - HTML/XML entity escaping
    - WSL to Windows path conversion
    - Path separator normalization
    - Special characters in paths
    - Relative path resolution
    
    Args:
        path (str): The path to normalize
        
    Returns:
        str: The normalized path for the current OS
    """
    if not path:
        return path
        
    # Step 1: Decode URL-encoded characters
    path = unquote(path)
    
    # Step 2: Remove HTML/XML entity artifacts
    path = re.sub(r'&[^&]*?;', lambda m: m.group(0).replace(';', ''), path)
    
    # Step 3: Convert WSL paths to Windows if needed
    if path.startswith('/mnt/') and len(path) > 6 and path[5] in 'abcdefghijklmnopqrstuvwxyz':
        # /mnt/c/... -> c:/...
        path = f"{path[5]}:/{path[7:]}"
    
    # Step 4: Convert to Path object for normalization
    try:
        path_obj = Path(path)
        # Resolve the path to handle .. and . components
        if path_obj.exists():
            path_obj = path_obj.resolve()
        # Convert to string using proper separators for current OS
        path = str(path_obj)
    except Exception:
        # If Path conversion fails, just normalize slashes
        path = path.replace('\\', os.sep).replace('/', os.sep)
    
    return path

def get_base_storage_path() -> Path:
    """
    Get the configured base storage path from settings.
    Creates the directory if it doesn't exist.
    
    Returns:
        Path: Path object for the base storage directory
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # We'll log a critical error if no configuration source can be found
    config_found = False
    
    try:
        # Directly try to load from YAML file which is our primary source of truth
        try:
            import yaml
            # Try multiple possible locations for config.yaml
            possible_config_paths = [
                Path(__file__).parents[3] / 'config.yaml',  # /project/config.yaml
                Path(__file__).parents[2] / 'config.yaml',  # /project/app/config.yaml
                Path(__file__).parent.parent.parent.parent / 'video-analyzer' / 'config.yaml',  # For WSL paths
            ]
            
            config_path = None
            for path in possible_config_paths:
                if path.exists():
                    config_path = path
                    logger.info(f"Found config file at: {config_path}")
                    break
                    
            if config_path and config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                if 'storage' in yaml_config and 'base_path' in yaml_config['storage']:
                    base_path = Path(yaml_config['storage']['base_path'])
                    logger.info(f"Using base_path from config.yaml: {base_path}")
                    config_found = True
                else:
                    logger.error("storage.base_path not found in config.yaml")
            else:
                logger.error(f"Config file not found at {config_path}")
        except Exception as yaml_err:
            logger.error(f"Error loading config.yaml: {str(yaml_err)}")
        
        # Only try other methods if direct YAML loading failed
        if not config_found:
            # Try the unified config system with direct attribute access
            try:
                from app.core.unified_config import get_config
                config = get_config()
                
                if hasattr(config, 'storage') and hasattr(config.storage, 'base_path'):
                    base_path = Path(config.storage.base_path)
                    logger.info(f"Using base_path from unified config: {base_path}")
                    config_found = True
            except Exception as unified_err:
                logger.warning(f"Could not access base_path from unified config: {str(unified_err)}")
            
            # If still not found, try legacy settings
            if not config_found:
                try:
                    from app.core.config import get_settings
                    settings = get_settings()
                    if hasattr(settings, 'storage') and hasattr(settings.storage, 'base_path'):
                        base_path = Path(settings.storage.base_path)
                        logger.info(f"Using base_path from legacy settings: {base_path}")
                        config_found = True
                except Exception as settings_err:
                    logger.warning(f"Could not access base_path from legacy settings: {str(settings_err)}")
    
    except Exception as e:
        logger.error(f"Unexpected error determining base storage path: {str(e)}")
    
    # If we couldn't find any configuration, log a critical error
    if not config_found:
        logger.critical("No storage.base_path configuration found in any config source!")
        logger.critical("Please ensure storage.base_path is properly set in config.yaml")
        raise ValueError("storage.base_path not configured - application cannot proceed without explicit configuration")
    
    # Ensure the directory exists
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        
        if base_path.exists():
            logger.info(f"Successfully verified/created base storage directory at {base_path}")
        else:
            logger.critical(f"Failed to create directory at {base_path} - directory does not exist after creation attempt")
            raise IOError(f"Could not create directory at {base_path}")
    except Exception as mkdir_err:
        logger.critical(f"Critical error creating directory at {base_path}: {str(mkdir_err)}")
        raise IOError(f"Failed to create storage directory: {str(mkdir_err)}")
        
    return base_path

def get_storage_subdir(subdir_name: str) -> Path:
    """
    Get a subdirectory within the base storage path.
    Creates the directory if it doesn't exist.
    
    Args:
        subdir_name (str): Name of the subdirectory
        
    Returns:
        Path: Path object for the subdirectory
    """
    base_path = get_base_storage_path()
    subdir_path = base_path / subdir_name
    subdir_path.mkdir(parents=True, exist_ok=True)
    return subdir_path

def get_storage_path(relative_path: str) -> Path:
    """
    Get a path within the base storage directory.
    Creates parent directories if they don't exist.
    
    Args:
        relative_path (str): Path relative to the base storage directory
        
    Returns:
        Path: Absolute path within the storage directory
    """
    base_path = get_base_storage_path()
    full_path = base_path / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path

def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path (Union[str, Path]): Directory path to ensure exists
        
    Returns:
        Path: Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
