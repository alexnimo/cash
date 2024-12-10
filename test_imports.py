import sys
import os
from pathlib import Path

# Get the absolute path to the project root directory
project_root = str(Path(__file__).parent.absolute())
print(f"Project root: {project_root}")

# Add the project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"\nPython path:")
for path in sys.path:
    print(f"  - {path}")

print("\nChecking app directory structure:")
app_dir = os.path.join(project_root, "app")
print(f"app directory exists: {os.path.exists(app_dir)}")
print(f"app/__init__.py exists: {os.path.exists(os.path.join(app_dir, '__init__.py'))}")

print("\nTrying imports one by one:")
try:
    import app
    print("✓ Successfully imported app")
    
    import app.core
    print("✓ Successfully imported app.core")
    
    from app.core import config
    print("✓ Successfully imported app.core.config")
    
    from app.core.config import get_settings
    print("✓ Successfully imported get_settings")
    
except ImportError as e:
    print(f"✗ Import failed: {str(e)}")
    print(f"Failed module's __file__: {getattr(sys.modules.get(e.name, None), '__file__', 'N/A')}")
    print(f"Failed module's __path__: {getattr(sys.modules.get(e.name, None), '__path__', 'N/A')}")
