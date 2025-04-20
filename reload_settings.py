"""
Utility script to reload settings from config.yaml file.
Run this script after making changes to config.yaml to apply the changes
without having to restart the entire application.
"""
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

if __name__ == "__main__":
    try:
        print("Reloading settings from config.yaml...")
        
        # Import the reload_settings function
        from app.core.settings import reload_settings
        from app.utils.path_utils import ensure_proper_path
        from pathlib import Path
        import sys
        
        # Reload the settings
        settings = reload_settings()
        
        # Get project root for making absolute paths
        project_root = Path(__file__).parent.absolute()
        
        # Handle the base storage directory
        base_storage_dir = ensure_proper_path(settings.storage.base_dir)
        if not base_storage_dir.is_absolute():
            base_storage_dir = project_root / base_storage_dir
        base_storage_dir = base_storage_dir.resolve()
        
        # Print storage configuration
        print("\nStorage configuration:")
        print(f"Base storage directory: {base_storage_dir}")
        print(f"Max cache size: {settings.storage.max_cache_size} bytes")
        
        # Print the subdirectories that will be created
        print("\nSubdirectories that will be created:")
        print(f"Videos directory: {base_storage_dir / 'videos'}")
        print(f"Audio directory: {base_storage_dir / 'audio'}")
        print(f"Transcripts directory: {base_storage_dir / 'transcripts'}")
        print(f"Raw transcripts directory: {base_storage_dir / 'raw_transcripts'}")
        print(f"Frames directory: {base_storage_dir / 'frames'}")
        print(f"Summaries directory: {base_storage_dir / 'summaries'}")
        print(f"Reports directory: {base_storage_dir / 'reports'}")
        print(f"Temp directory: {base_storage_dir / 'temp'}")
        print(f"Cache directory: {base_storage_dir / 'cache'}")
        
        # Check if settings will be used correctly in both Windows and WSL
        print("\nPath environment check:")
        print(f"Platform: {sys.platform}")
        if sys.platform == 'linux':
            print("Running in Linux/WSL environment")
            print("Ensure Windows drives are properly mounted in WSL")
            print("Check that /etc/wsl.conf has correct automount settings")
        else:
            print("Running in Windows environment")
        
        print("\nSettings reloaded successfully!")
        print("You may need to restart any components that have already loaded the old settings.")
        
    except Exception as e:
        print(f"Error reloading settings: {str(e)}")
        import traceback
        traceback.print_exc()
