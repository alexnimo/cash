"""
Pytest configuration and shared fixtures for the entire test suite.

Provides common test fixtures, configuration, and utilities for testing
the DannyTrades agent and other components.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from fastapi.testclient import TestClient

# Import the main app
from main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """Create a test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    test_env = {
        'TESTING': 'true',
        'NOTION_API_KEY': 'test-notion-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key',
        'OLLAMA_ENDPOINT': 'http://localhost:11434'
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def sample_stock_data():
    """Standard sample stock data for testing."""
    return {
        "symbol": "AAPL",
        "ticker_name": "Apple Inc.",
        "volatility_hole": True,
        "red_candle": {
            "daily": True,
            "weekly": False,
            "monthly": True
        },
        "yellow_candle": {
            "daily": False,
            "weekly": True,
            "monthly": False
        },
        "whale_accumulation": {
            "daily": "Strong",
            "weekly": "Moderate",
            "monthly": "Weak"
        },
        "technical_indicators": {
            "support_levels": [145.0, 140.0, 135.0],
            "resistance_levels": [155.0, 160.0, 165.0]
        },
        "bullishness_signal": ["Extreme Bullish"],
        "misc_data": "Strong institutional buying detected",
        "patreon_link": "https://patreon.com/dannytrades"
    }


@pytest.fixture
def mock_notion_mcp_tools():
    """Mock Notion MCP tools for testing."""
    tools = Mock()
    tools.query_notion_database = AsyncMock()
    tools.create_notion_page = AsyncMock()
    tools.update_notion_page = AsyncMock()
    tools.get_page = AsyncMock()
    return tools


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    response = Mock()
    response.content = '{"action": "update", "reasoning": "Test reasoning", "confidence": 0.95}'
    return response


@pytest.fixture
def test_data_dir():
    """Get the test data directory path."""
    return Path(__file__).parent / "data"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file location
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_agent" in item.nodeid or "test_models" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_mock_agent():
        """Create a mock DannyTrades agent."""
        agent = Mock()
        agent.initialized = True
        agent.current_provider = 'claude'
        agent.process_stock_data = AsyncMock()
        agent.get_status = AsyncMock()
        agent.get_activities = AsyncMock()
        agent.switch_provider = AsyncMock()
        agent.update_prompts = AsyncMock()
        return agent
    
    @staticmethod
    def create_processing_result(success=True, **kwargs):
        """Create a ProcessingResult for testing."""
        from app.agents.dannytrades.models import ProcessingResult
        
        defaults = {
            'success': success,
            'processing_time': 1.0
        }
        
        if success:
            defaults.update({
                'action': 'update',
                'reasoning': 'Test processing successful',
                'confidence': 0.9
            })
        else:
            defaults.update({
                'error_message': 'Test error occurred'
            })
        
        defaults.update(kwargs)
        return ProcessingResult(**defaults)


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils
