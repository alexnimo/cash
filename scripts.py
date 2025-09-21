#!/usr/bin/env python3
"""
Convenience scripts for the video-analyzer project using uv.
Usage: uv run python scripts.py <command>
"""
import sys
import subprocess
import os


def run_dev():
    """Run the application in development mode with auto-reload."""
    subprocess.run([
        "uv", "run", "uvicorn", "app.main:app", 
        "--reload", "--host", "0.0.0.0", "--port", "8000"
    ])


def run_start():
    """Run the application in production mode."""
    subprocess.run([
        "uv", "run", "uvicorn", "app.main:app", 
        "--host", "0.0.0.0", "--port", "8000"
    ])


def run_test():
    """Run tests."""
    subprocess.run(["uv", "run", "pytest", "tests/", "-v"])


def run_format():
    """Format code with black and isort."""
    subprocess.run(["uv", "run", "black", "."])
    subprocess.run(["uv", "run", "isort", "."])


def run_lint():
    """Run linting with flake8 and mypy."""
    subprocess.run(["uv", "run", "flake8", "."])
    subprocess.run(["uv", "run", "mypy", "."])


def setup_system():
    """Install system dependencies like ffmpeg."""
    subprocess.run(["uv", "run", "python", "setup.py", "install"])


def show_help():
    """Show available commands."""
    print("Available commands:")
    print("  dev      - Run in development mode with auto-reload")
    print("  start    - Run in production mode")
    print("  test     - Run tests")
    print("  format   - Format code with black and isort")
    print("  lint     - Run linting with flake8 and mypy")
    print("  setup    - Install system dependencies")
    print("  help     - Show this help message")


def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1]
    
    commands = {
        "dev": run_dev,
        "start": run_start,
        "test": run_test,
        "format": run_format,
        "lint": run_lint,
        "setup": setup_system,
        "help": show_help,
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
