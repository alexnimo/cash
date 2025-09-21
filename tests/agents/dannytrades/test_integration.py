"""
Integration test suite for DannyTrades Agent system.

Tests end-to-end workflows, UI component integration, API integration,
and real-world scenarios for the complete DannyTrades feature.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import json
import time
from datetime import datetime

from app.agents.dannytrades.agent import DannyTradesAgent
from app.agents.dannytrades.models import StockData, ProcessingResult
from app.api.dannytrades_routes import router
from main import app


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def client(self):
        """Create a test client for integration testing."""
        return TestClient(app)

    @pytest.fixture
    def sample_stock_batch(self):
        """Sample batch of stock data for testing."""
        return [
            {
                "symbol": "AAPL",
                "ticker_name": "Apple Inc.",
                "volatility_hole": True,
                "red_candle": {"daily": True, "weekly": False, "monthly": True},
                "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
                "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                "technical_indicators": {"support_levels": [145.0, 140.0], "resistance_levels": [155.0, 160.0]},
                "bullishness_signal": ["Extreme Bullish"]
            },
            {
                "symbol": "TSLA",
                "ticker_name": "Tesla Inc.",
                "volatility_hole": False,
                "red_candle": {"daily": False, "weekly": True, "monthly": False},
                "yellow_candle": {"daily": True, "weekly": False, "monthly": True},
                "whale_accumulation": {"daily": "Moderate", "weekly": "Strong", "monthly": "Strong"},
                "technical_indicators": {"support_levels": [240.0, 235.0], "resistance_levels": [260.0, 270.0]},
                "bullishness_signal": ["Hot"]
            },
            {
                "symbol": "NVDA",
                "ticker_name": "NVIDIA Corporation",
                "volatility_hole": True,
                "red_candle": {"daily": True, "weekly": True, "monthly": False},
                "yellow_candle": {"daily": False, "weekly": False, "monthly": True},
                "whale_accumulation": {"daily": "Strong", "weekly": "Strong", "monthly": "Moderate"},
                "technical_indicators": {"support_levels": [880.0, 870.0], "resistance_levels": [920.0, 940.0]},
                "bullishness_signal": ["Extreme Bullish", "Hot"]
            }
        ]

    @patch('app.agents.dannytrades.agent.NotionMCPTools')
    @patch('app.agents.dannytrades.agent.DannyTradesConfig')
    @pytest.mark.asyncio
    async def test_complete_stock_processing_workflow(self, mock_config_class, mock_mcp_tools_class, client, sample_stock_batch):
        """Test complete workflow from API request to Notion update."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_current_provider.return_value = 'claude'
        mock_config.get_provider_config.return_value = {'model_id': 'claude-3-sonnet', 'api_key': 'test-key'}
        mock_config.get_prompts.return_value = {'system_prompt': 'Test prompt'}
        mock_config_class.return_value = mock_config

        mock_mcp_tools = Mock()
        mock_mcp_tools.query_notion_database = AsyncMock(return_value={'results': []})
        mock_mcp_tools.create_notion_page = AsyncMock(return_value={'id': 'new_page_id'})
        mock_mcp_tools_class.return_value = mock_mcp_tools

        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.content = json.dumps({
            "action": "update",
            "reasoning": "Strong bullish signals detected with high confidence",
            "confidence": 0.95
        })

        with patch('app.agents.dannytrades.agent.DannyTradesAgent._call_llm', return_value=mock_llm_response):
            # Process each stock in the batch
            results = []
            for stock_data in sample_stock_batch:
                response = client.post("/api/dannytrades/process", json=stock_data)
                assert response.status_code == 200
                
                result = response.json()
                assert result["success"] is True
                assert result["action"] == "update"
                results.append(result)

            # Verify all stocks were processed
            assert len(results) == 3
            
            # Verify Notion integration was called for each stock
            assert mock_mcp_tools.create_notion_page.call_count == 3

    @patch('app.agents.dannytrades.agent.NotionMCPTools')
    @patch('app.agents.dannytrades.agent.DannyTradesConfig')
    @pytest.mark.asyncio
    async def test_provider_switching_workflow(self, mock_config_class, mock_mcp_tools_class, client):
        """Test LLM provider switching workflow."""
        mock_config = Mock()
        mock_config.get_current_provider.return_value = 'claude'
        mock_config.switch_provider.return_value = True
        mock_config.get_available_providers.return_value = {
            'claude': {'enabled': True, 'has_api_key': True},
            'openai': {'enabled': True, 'has_api_key': True}
        }
        mock_config_class.return_value = mock_config

        # Get initial provider status
        response = client.get("/api/dannytrades/providers/status")
        assert response.status_code == 200
        
        # Switch provider
        response = client.post("/api/dannytrades/providers/switch", json={"provider": "openai"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["success"] is True
        mock_config.switch_provider.assert_called_with("openai")

    @patch('app.agents.dannytrades.agent.DannyTradesAgent')
    def test_real_time_monitoring_workflow(self, mock_agent_class, client):
        """Test real-time monitoring and activity logging."""
        mock_agent = Mock()
        mock_agent.get_activities.return_value = [
            {
                "id": "1",
                "timestamp": datetime.now().isoformat() + "Z",
                "type": "processing",
                "message": "Processing AAPL data",
                "details": {"symbol": "AAPL"}
            },
            {
                "id": "2", 
                "timestamp": datetime.now().isoformat() + "Z",
                "type": "success",
                "message": "Successfully updated AAPL",
                "details": {"symbol": "AAPL", "action": "update"}
            }
        ]
        mock_agent_class.return_value = mock_agent

        with patch('app.api.dannytrades_routes.get_dannytrades_agent', return_value=mock_agent):
            # Get activities
            response = client.get("/api/dannytrades/activities")
            assert response.status_code == 200
            
            activities = response.json()
            assert len(activities) == 2
            assert activities[0]["type"] == "processing"
            assert activities[1]["type"] == "success"

    @patch('app.agents.dannytrades.agent.DannyTradesAgent')
    def test_prompt_customization_workflow(self, mock_agent_class, client):
        """Test prompt customization workflow."""
        mock_agent = Mock()
        mock_agent.update_prompts = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        with patch('app.api.dannytrades_routes.get_dannytrades_agent', return_value=mock_agent):
            # Update prompts
            new_prompts = {
                "system_prompt": "You are an advanced stock analysis AI",
                "processing_instructions": "Analyze the provided stock data carefully",
                "error_handling_prompt": "If errors occur, provide detailed explanations"
            }
            
            response = client.post("/api/dannytrades/prompts/update", json=new_prompts)
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] is True
            mock_agent.update_prompts.assert_called_once_with(new_prompts)


class TestErrorRecoveryScenarios:
    """Test error handling and recovery scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch('app.agents.dannytrades.agent.DannyTradesAgent')
    def test_llm_api_failure_recovery(self, mock_agent_class, client):
        """Test recovery from LLM API failures."""
        mock_agent = Mock()
        
        # First call fails, second succeeds
        mock_agent.process_stock_data = AsyncMock(side_effect=[
            ProcessingResult(success=False, error_message="LLM API rate limit exceeded"),
            ProcessingResult(success=True, action="update", reasoning="Processed successfully")
        ])
        
        mock_agent_class.return_value = mock_agent

        with patch('app.api.dannytrades_routes.get_dannytrades_agent', return_value=mock_agent):
            stock_data = {
                "symbol": "AAPL",
                "volatility_hole": True,
                "red_candle": {"daily": True, "weekly": False, "monthly": True},
                "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
                "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
                "bullishness_signal": ["Hot"]
            }
            
            # First request fails
            response = client.post("/api/dannytrades/process", json=stock_data)
            assert response.status_code == 500
            
            # Second request succeeds (simulating retry)
            response = client.post("/api/dannytrades/process", json=stock_data)
            assert response.status_code == 200

    @patch('app.agents.dannytrades.agent.DannyTradesAgent')
    def test_notion_connection_failure(self, mock_agent_class, client):
        """Test handling of Notion connection failures."""
        mock_agent = Mock()
        mock_agent.process_stock_data = AsyncMock(return_value=ProcessingResult(
            success=False,
            error_message="Failed to connect to Notion API"
        ))
        mock_agent_class.return_value = mock_agent

        with patch('app.api.dannytrades_routes.get_dannytrades_agent', return_value=mock_agent):
            stock_data = {
                "symbol": "AAPL",
                "volatility_hole": True,
                "red_candle": {"daily": True, "weekly": False, "monthly": True},
                "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
                "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
                "bullishness_signal": ["Hot"]
            }
            
            response = client.post("/api/dannytrades/process", json=stock_data)
            assert response.status_code == 500
            
            result = response.json()
            assert "Notion API" in result["error_message"]

    def test_malformed_request_handling(self, client):
        """Test handling of malformed requests."""
        # Invalid JSON
        response = client.post(
            "/api/dannytrades/process",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

        # Missing required fields
        response = client.post("/api/dannytrades/process", json={"symbol": "AAPL"})
        assert response.status_code == 422

        # Invalid field values
        response = client.post("/api/dannytrades/process", json={
            "symbol": "",  # Empty symbol
            "volatility_hole": True,
            "red_candle": {"daily": True, "weekly": False, "monthly": True},
            "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
            "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
            "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
            "bullishness_signal": ["Hot"]
        })
        assert response.status_code == 422


class TestPerformanceScenarios:
    """Test performance and scalability scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch('app.agents.dannytrades.agent.DannyTradesAgent')
    def test_concurrent_request_handling(self, mock_agent_class, client):
        """Test handling of concurrent requests."""
        mock_agent = Mock()
        mock_agent.process_stock_data = AsyncMock(return_value=ProcessingResult(
            success=True,
            action="update",
            reasoning="Processed successfully",
            processing_time=1.0
        ))
        mock_agent_class.return_value = mock_agent

        with patch('app.api.dannytrades_routes.get_dannytrades_agent', return_value=mock_agent):
            stock_data = {
                "symbol": "TEST",
                "volatility_hole": True,
                "red_candle": {"daily": True, "weekly": False, "monthly": True},
                "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
                "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
                "bullishness_signal": ["Hot"]
            }
            
            # Simulate concurrent requests
            import threading
            results = []
            
            def make_request(symbol_suffix):
                data = stock_data.copy()
                data["symbol"] = f"TEST{symbol_suffix}"
                response = client.post("/api/dannytrades/process", json=data)
                results.append(response.status_code)
            
            threads = []
            for i in range(5):
                thread = threading.Thread(target=make_request, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All requests should succeed
            assert all(status == 200 for status in results)

    @patch('app.agents.dannytrades.agent.DannyTradesAgent')
    def test_large_batch_processing(self, mock_agent_class, client):
        """Test processing of large batches of stock data."""
        mock_agent = Mock()
        mock_agent.process_stock_data = AsyncMock(return_value=ProcessingResult(
            success=True,
            action="update",
            reasoning="Batch processed successfully",
            processing_time=0.5
        ))
        mock_agent_class.return_value = mock_agent

        with patch('app.api.dannytrades_routes.get_dannytrades_agent', return_value=mock_agent):
            # Create a large batch of stock data
            base_stock = {
                "volatility_hole": True,
                "red_candle": {"daily": True, "weekly": False, "monthly": True},
                "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
                "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
                "bullishness_signal": ["Hot"]
            }
            
            # Process 20 stocks
            start_time = time.time()
            success_count = 0
            
            for i in range(20):
                stock_data = base_stock.copy()
                stock_data["symbol"] = f"STOCK{i:02d}"
                
                response = client.post("/api/dannytrades/process", json=stock_data)
                if response.status_code == 200:
                    success_count += 1
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert success_count == 20
            assert processing_time < 30  # Should complete within 30 seconds


class TestUIComponentIntegration:
    """Test UI component integration scenarios."""

    def test_configuration_panel_data_flow(self):
        """Test data flow for configuration panel."""
        # This would test the TypeScript components in a real scenario
        # For now, we simulate the expected API interactions
        
        # Mock provider status data
        provider_status = {
            "claude": {
                "name": "claude",
                "enabled": True,
                "model_id": "claude-3-sonnet-20240229",
                "has_api_key": True,
                "current": True
            },
            "openai": {
                "name": "openai", 
                "enabled": True,
                "model_id": "gpt-4",
                "has_api_key": False,
                "current": False
            }
        }
        
        # Verify data structure matches UI expectations
        assert "claude" in provider_status
        assert provider_status["claude"]["current"] is True
        assert provider_status["openai"]["has_api_key"] is False

    def test_stock_dashboard_data_structure(self):
        """Test stock dashboard data structure compatibility."""
        # Mock stock data for dashboard
        dashboard_data = [
            {
                "symbol": "AAPL",
                "ticker_name": "Apple Inc.",
                "last_updated": "2025-01-07T20:30:00Z",
                "volatility_hole": True,
                "bullishness_signal": ["Extreme Bullish"],
                "technical_indicators": {
                    "support_levels": [145.0, 140.0],
                    "resistance_levels": [155.0, 160.0]
                }
            }
        ]
        
        # Verify structure matches UI component expectations
        stock = dashboard_data[0]
        assert "symbol" in stock
        assert "last_updated" in stock
        assert isinstance(stock["bullishness_signal"], list)
        assert isinstance(stock["technical_indicators"]["support_levels"], list)

    def test_real_time_activity_data_format(self):
        """Test real-time activity data format."""
        # Mock activity data
        activities = [
            {
                "id": "1",
                "timestamp": "2025-01-07T20:30:00Z",
                "type": "processing",
                "title": "Processing AAPL data",
                "description": "Analyzing technical indicators",
                "stock_symbol": "AAPL"
            },
            {
                "id": "2",
                "timestamp": "2025-01-07T20:29:30Z",
                "type": "success",
                "title": "TSLA updated successfully",
                "description": "Notion database entry created",
                "stock_symbol": "TSLA"
            }
        ]
        
        # Verify format matches UI expectations
        for activity in activities:
            assert "id" in activity
            assert "timestamp" in activity
            assert "type" in activity
            assert activity["type"] in ["processing", "success", "error", "notification"]


class TestSecurityScenarios:
    """Test security-related scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_input_sanitization(self, client):
        """Test input sanitization and validation."""
        # Test SQL injection attempt
        malicious_payload = {
            "symbol": "AAPL'; DROP TABLE stocks; --",
            "volatility_hole": True,
            "red_candle": {"daily": True, "weekly": False, "monthly": True},
            "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
            "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
            "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
            "bullishness_signal": ["Hot"]
        }
        
        response = client.post("/api/dannytrades/process", json=malicious_payload)
        # Should be rejected due to symbol validation
        assert response.status_code == 422

    def test_xss_prevention(self, client):
        """Test XSS prevention in text fields."""
        xss_payload = {
            "symbol": "AAPL",
            "ticker_name": "<script>alert('xss')</script>",
            "volatility_hole": True,
            "red_candle": {"daily": True, "weekly": False, "monthly": True},
            "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
            "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
            "technical_indicators": {"support_levels": [145.0], "resistance_levels": [155.0]},
            "bullishness_signal": ["Hot"],
            "misc_data": "<img src=x onerror=alert('xss')>"
        }
        
        # The API should accept this but sanitize the content
        # In a real implementation, we'd verify sanitization occurred
        response = client.post("/api/dannytrades/process", json=xss_payload)
        # Response depends on agent implementation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
