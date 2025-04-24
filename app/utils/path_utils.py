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
    from app.core.config import get_settings
    
    base_path = Path(get_settings().storage.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
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
