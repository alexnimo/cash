# UV Migration Guide

This project has been migrated from pip/conda to use **uv** for faster and more reliable dependency management.

## Quick Start

### 1. Install dependencies
```bash
uv sync --all-extras
```

### 2. Run the application
```bash
# Development mode with auto-reload
uv run python scripts.py dev

# Production mode
uv run python scripts.py start

# Or directly:
uv run uvicorn app.main:app --reload
```

### 3. Install system dependencies (FFmpeg)
```bash
uv run python scripts.py setup
```

## Available Commands

### Using the scripts helper
```bash
uv run python scripts.py <command>
```

Available commands:
- `dev` - Run in development mode with auto-reload
- `start` - Run in production mode
- `test` - Run tests
- `format` - Format code with black and isort
- `lint` - Run linting with flake8 and mypy
- `setup` - Install system dependencies

### Direct uv commands
```bash
# Install dependencies
uv sync                    # Install all dependencies
uv sync --no-dev          # Install only production dependencies
uv sync --all-extras      # Install all optional dependencies

# Add/remove dependencies
uv add package-name       # Add new dependency
uv remove package-name    # Remove dependency

# Run commands in the virtual environment
uv run python main.py     # Run any Python script
uv run pytest tests/      # Run tests
uv run black .            # Format code
```

## Migration Benefits

- **10-100x faster** dependency resolution and installation
- **Better dependency resolution** - more reliable than pip
- **Built-in virtual environment management**
- **Lock file** (`uv.lock`) for reproducible builds
- **No need for conda** - uv handles everything
- **Cross-platform compatibility**

## Project Structure

- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Dependency lock file (don't edit manually)
- `scripts.py` - Convenience scripts for common tasks
- `.venv/` - Virtual environment (created by uv)

## Troubleshooting

### If you get import errors:
```bash
uv sync --all-extras
```

### If FFmpeg is not found:
```bash
uv run python scripts.py setup
```

### To recreate the virtual environment:
```bash
rm -rf .venv
uv sync --all-extras
```

## Dependency Management

### Add a new dependency:
```bash
uv add requests>=2.28.0
```

### Add a development dependency:
```bash
uv add --dev pytest-coverage
```

### Remove a dependency:
```bash
uv remove requests
```

The `pyproject.toml` file will be automatically updated, and you should commit both `pyproject.toml` and `uv.lock` to version control.
