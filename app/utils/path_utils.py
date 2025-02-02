import os
import re
from pathlib import Path
from urllib.parse import unquote

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
