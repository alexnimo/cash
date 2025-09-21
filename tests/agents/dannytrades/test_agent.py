"""
Test suite for DannyTrades Agent functionality.

Tests the core agent logic, LLM provider switching, MCP integration,
and stock data processing workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from app.agents.dannytrades.agent import DannyTradesAgent
from app.agents.dannytrades.models import StockData, ProcessingResult
from app.agents.dannytrades.config import DannyTradesConfig


class TestDannyTradesAgent:
    """Test cases for DannyTrades Agent core functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=DannyTradesConfig)
        config.get_current_provider.return_value = 'claude'
        config.get_provider_config.return_value = {
            'model_id': 'claude-3-sonnet-20240229',
            'api_key': 'test-key'
        }
        config.get_prompts.return_value = {
            'system_prompt': 'Test system prompt',
            'processing_instructions': 'Test instructions',
            'error_handling_prompt': 'Test error handling'
        }
        return config

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock data for testing."""
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
            "misc_data": "Strong institutional buying detected"
        }

    @pytest.fixture
    async def agent(self, mock_config):
        """Create a DannyTrades agent instance for testing."""
        with patch('app.agents.dannytrades.agent.DannyTradesConfig', return_value=mock_config):
            agent = DannyTradesAgent()
            await agent.initialize()
            return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_config):
        """Test agent initialization process."""
        with patch('app.agents.dannytrades.agent.DannyTradesConfig', return_value=mock_config):
            agent = DannyTradesAgent()
            
            # Test initialization
            await agent.initialize()
            
            assert agent.initialized is True
            assert agent.current_provider == 'claude'
            mock_config.get_current_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_stock_data_success(self, agent, sample_stock_data):
        """Test successful stock data processing."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "action": "update",
            "reasoning": "Strong bullish signals detected",
            "confidence": 0.95
        })

        with patch.object(agent, '_call_llm', return_value=mock_response):
            with patch.object(agent, '_update_notion_database', return_value=True) as mock_update:
                result = await agent.process_stock_data(sample_stock_data)
                
                assert isinstance(result, ProcessingResult)
                assert result.success is True
                assert result.action == "update"
                assert "Strong bullish signals" in result.reasoning
                mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_stock_data_validation_error(self, agent):
        """Test stock data processing with invalid input."""
        invalid_data = {"symbol": "AAPL"}  # Missing required fields
        
        result = await agent.process_stock_data(invalid_data)
        
        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert "validation" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_process_stock_data_llm_error(self, agent, sample_stock_data):
        """Test stock data processing with LLM error."""
        with patch.object(agent, '_call_llm', side_effect=Exception("LLM API error")):
            result = await agent.process_stock_data(sample_stock_data)
            
            assert isinstance(result, ProcessingResult)
            assert result.success is False
            assert "LLM API error" in result.error_message

    @pytest.mark.asyncio
    async def test_switch_provider_success(self, agent, mock_config):
        """Test successful LLM provider switching."""
        mock_config.switch_provider.return_value = True
        mock_config.get_current_provider.return_value = 'openai'
        
        result = await agent.switch_provider('openai')
        
        assert result is True
        assert agent.current_provider == 'openai'
        mock_config.switch_provider.assert_called_once_with('openai')

    @pytest.mark.asyncio
    async def test_switch_provider_failure(self, agent, mock_config):
        """Test failed LLM provider switching."""
        mock_config.switch_provider.return_value = False
        
        result = await agent.switch_provider('invalid_provider')
        
        assert result is False
        mock_config.switch_provider.assert_called_once_with('invalid_provider')

    @pytest.mark.asyncio
    async def test_update_prompts(self, agent, mock_config):
        """Test prompt updating functionality."""
        new_prompts = {
            'system_prompt': 'Updated system prompt',
            'processing_instructions': 'Updated instructions'
        }
        
        mock_config.update_prompts.return_value = True
        
        result = await agent.update_prompts(new_prompts)
        
        assert result is True
        mock_config.update_prompts.assert_called_once_with(new_prompts)

    @pytest.mark.asyncio
    async def test_get_activities(self, agent):
        """Test activity log retrieval."""
        # Add some test activities
        await agent._log_activity('processing', 'Test processing', {'symbol': 'AAPL'})
        await agent._log_activity('success', 'Test success', {'symbol': 'TSLA'})
        
        activities = await agent.get_activities(limit=10)
        
        assert len(activities) == 2
        assert activities[0]['type'] == 'success'  # Most recent first
        assert activities[1]['type'] == 'processing'

    @pytest.mark.asyncio
    async def test_get_status(self, agent):
        """Test agent status retrieval."""
        status = await agent.get_status()
        
        assert 'initialized' in status
        assert 'current_provider' in status
        assert 'last_activity' in status
        assert status['initialized'] is True

    def test_stock_data_model_validation(self, sample_stock_data):
        """Test StockData model validation."""
        # Valid data
        stock_data = StockData(**sample_stock_data)
        assert stock_data.symbol == "AAPL"
        assert stock_data.volatility_hole is True
        
        # Invalid data - missing required field
        invalid_data = sample_stock_data.copy()
        del invalid_data['symbol']
        
        with pytest.raises(ValueError):
            StockData(**invalid_data)

    def test_processing_result_model(self):
        """Test ProcessingResult model creation."""
        result = ProcessingResult(
            success=True,
            action="update",
            reasoning="Test reasoning",
            confidence=0.95,
            processing_time=2.5
        )
        
        assert result.success is True
        assert result.action == "update"
        assert result.confidence == 0.95


class TestDannyTradesConfig:
    """Test cases for DannyTrades configuration management."""

    @pytest.fixture
    def config(self):
        """Create a configuration instance for testing."""
        return DannyTradesConfig()

    def test_get_available_providers(self, config):
        """Test getting available LLM providers."""
        providers = config.get_available_providers()
        
        assert isinstance(providers, dict)
        assert 'claude' in providers
        assert 'openai' in providers
        
        # Check provider structure
        claude_config = providers['claude']
        assert 'model_id' in claude_config
        assert 'enabled' in claude_config

    def test_provider_validation(self, config):
        """Test LLM provider validation."""
        # Valid providers
        assert config._validate_provider('claude') is True
        assert config._validate_provider('openai') is True
        
        # Invalid provider
        assert config._validate_provider('invalid') is False

    def test_prompt_management(self, config):
        """Test prompt getting and updating."""
        # Get default prompts
        prompts = config.get_prompts()
        assert isinstance(prompts, dict)
        assert 'system_prompt' in prompts
        
        # Update prompts
        new_prompts = {'system_prompt': 'Updated prompt'}
        result = config.update_prompts(new_prompts)
        assert result is True
        
        # Verify update
        updated_prompts = config.get_prompts()
        assert updated_prompts['system_prompt'] == 'Updated prompt'

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    def test_api_key_detection(self, config):
        """Test API key detection from environment."""
        claude_config = config.get_provider_config('claude')
        assert claude_config['has_api_key'] is True

    @patch.dict('os.environ', {}, clear=True)
    def test_missing_api_key_detection(self, config):
        """Test detection of missing API keys."""
        claude_config = config.get_provider_config('claude')
        assert claude_config['has_api_key'] is False


class TestNotionMCPIntegration:
    """Test cases for Notion MCP integration."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        tools = Mock()
        tools.query_notion_database = AsyncMock()
        tools.create_notion_page = AsyncMock()
        tools.update_notion_page = AsyncMock()
        return tools

    @pytest.mark.asyncio
    async def test_notion_query(self, mock_mcp_tools):
        """Test Notion database querying."""
        # Mock query response
        mock_mcp_tools.query_notion_database.return_value = {
            'results': [
                {'id': 'page1', 'properties': {'Symbol': {'title': [{'text': {'content': 'AAPL'}}]}}}
            ]
        }
        
        with patch('app.agents.dannytrades.agent.NotionMCPTools', return_value=mock_mcp_tools):
            agent = DannyTradesAgent()
            await agent.initialize()
            
            result = await agent._query_existing_stock('AAPL')
            
            assert result is not None
            mock_mcp_tools.query_notion_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_notion_create_page(self, mock_mcp_tools):
        """Test Notion page creation."""
        mock_mcp_tools.create_notion_page.return_value = {'id': 'new_page_id'}
        
        with patch('app.agents.dannytrades.agent.NotionMCPTools', return_value=mock_mcp_tools):
            agent = DannyTradesAgent()
            await agent.initialize()
            
            stock_data = StockData(
                symbol="TSLA",
                volatility_hole=False,
                red_candle={"daily": False, "weekly": True, "monthly": False},
                yellow_candle={"daily": True, "weekly": False, "monthly": True},
                whale_accumulation={"daily": "Moderate", "weekly": "Strong", "monthly": "Strong"},
                technical_indicators={"support_levels": [240.0], "resistance_levels": [260.0]},
                bullishness_signal=["Hot"]
            )
            
            result = await agent._create_notion_page(stock_data)
            
            assert result is True
            mock_mcp_tools.create_notion_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_notion_update_page(self, mock_mcp_tools):
        """Test Notion page updating."""
        mock_mcp_tools.update_notion_page.return_value = {'id': 'updated_page_id'}
        
        with patch('app.agents.dannytrades.agent.NotionMCPTools', return_value=mock_mcp_tools):
            agent = DannyTradesAgent()
            await agent.initialize()
            
            result = await agent._update_notion_page('page_id', {'Symbol': 'AAPL'})
            
            assert result is True
            mock_mcp_tools.update_notion_page.assert_called_once_with('page_id', {'Symbol': 'AAPL'})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
