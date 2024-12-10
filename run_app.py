import os
import sys
from pathlib import Path

# Get the absolute path to the project root directory
project_root = str(Path(__file__).parent.absolute())

# Add the project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change working directory to project root
os.chdir(project_root)

try:
    import app
    print("Successfully imported app module")
    from app.services.model_manager import ModelManager
    from app.core.config import get_settings
    
    settings = get_settings()
    model_manager = ModelManager(settings)
    print("Successfully initialized ModelManager")
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    print(f"\nCurrent working directory: {os.getcwd()}")
