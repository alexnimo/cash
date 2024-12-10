#!/bin/bash

echo "🚀 Setting up Video Analyzer..."
echo "==============================="

# Function to check Python version
check_python_version() {
    echo "📋 Checking Python version..."
    python3 --version || {
        echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    }
}

# Function to verify project structure
verify_project_structure() {
    echo -e "\n📁 Verifying project structure..."
    
    # Check if running from project root
    if [ ! -f "setup.py" ] || [ ! -d "app" ]; then
        echo "❌ Please run this script from the project root directory"
        echo "   Current directory: $(pwd)"
        exit 1
    fi
    
    echo "✓ Running from correct directory"
    
    # Check for required files
    [ -f "requirements.txt" ] && echo "✓ Found requirements.txt" || echo "❌ Missing requirements.txt"
    [ -f "config.yaml" ] && echo "✓ Found config.yaml" || echo "❌ Missing config.yaml"
    [ -f ".env" ] && echo "✓ Found .env" || echo "❌ Missing .env"
}

# Function to create and activate virtual environment
setup_virtual_env() {
    echo -e "\n🔧 Setting up Python virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate || {
        echo "❌ Failed to activate virtual environment"
        exit 1
    }
    echo "✓ Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    echo -e "\n📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
}

# Function to verify imports
verify_imports() {
    echo -e "\n🔍 Verifying Python imports..."
    python3 - <<EOF
import sys
import os
from pathlib import Path

# Get the absolute path to the project root directory
project_root = str(Path().absolute())
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
    print(f"❌ Import failed: {str(e)}")
    print(f"Failed module's __file__: {getattr(sys.modules.get(e.name, None), '__file__', 'N/A')}")
    print(f"Failed module's __path__: {getattr(sys.modules.get(e.name, None), '__path__', 'N/A')}")
EOF
}

# Main setup process
check_python_version
verify_project_structure
setup_virtual_env
install_dependencies
verify_imports

echo -e "\n✨ Setup completed!"
echo "To start the server, run:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start the server: python -m app.main"
