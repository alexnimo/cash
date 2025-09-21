"""
DannyTrades Agent Test Suite

Comprehensive test suite for the DannyTrades agent feature including:
- Unit tests for agent logic and models
- API endpoint testing
- Integration testing
- Performance and security testing
"""

# Test configuration
import pytest
import asyncio
import os
from pathlib import Path

# Set test environment variables
os.environ['TESTING'] = 'true'
os.environ['NOTION_API_KEY'] = 'test-notion-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'

# Configure asyncio for testing
pytest_plugins = ['pytest_asyncio']
