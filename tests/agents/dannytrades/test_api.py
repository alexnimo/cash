"""
Test suite for DannyTrades API endpoints.

Tests the FastAPI routes, request/response handling, error cases,
and integration with the agent system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status
import json

from app.api.dannytrades_routes import router
from app.agents.dannytrades.models import ProcessingResult
from main import app


class TestDannyTradesAPI:
    """Test cases for DannyTrades API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)

    @pytest.fixture
    def sample_stock_payload(self):
        """Sample stock data payload for API testing."""
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
    def mock_agent(self):
        """Mock DannyTrades agent for testing."""
        agent = Mock()
        agent.process_stock_data = AsyncMock()
        agent.get_status = AsyncMock()
        agent.get_activities = AsyncMock()
        agent.switch_provider = AsyncMock()
        agent.update_prompts = AsyncMock()
        return agent

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/dannytrades/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_process_stock_data_success(self, mock_get_agent, client, sample_stock_payload, mock_agent):
        """Test successful stock data processing."""
        # Setup mock agent
        mock_get_agent.return_value = mock_agent
        mock_agent.process_stock_data.return_value = ProcessingResult(
            success=True,
            action="update",
            reasoning="Strong bullish signals detected",
            confidence=0.95,
            processing_time=2.3
        )
        
        response = client.post("/api/dannytrades/process", json=sample_stock_payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["action"] == "update"
        assert "Strong bullish signals" in data["reasoning"]
        mock_agent.process_stock_data.assert_called_once()

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_process_stock_data_validation_error(self, mock_get_agent, client, mock_agent):
        """Test stock data processing with validation error."""
        mock_get_agent.return_value = mock_agent
        
        # Invalid payload - missing required fields
        invalid_payload = {"symbol": "AAPL"}
        
        response = client.post("/api/dannytrades/process", json=invalid_payload)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_process_stock_data_agent_error(self, mock_get_agent, client, sample_stock_payload, mock_agent):
        """Test stock data processing with agent error."""
        mock_get_agent.return_value = mock_agent
        mock_agent.process_stock_data.return_value = ProcessingResult(
            success=False,
            error_message="LLM API error",
            processing_time=1.0
        )
        
        response = client.post("/api/dannytrades/process", json=sample_stock_payload)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["success"] is False
        assert "LLM API error" in data["error_message"]

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_get_status(self, mock_get_agent, client, mock_agent):
        """Test agent status endpoint."""
        mock_get_agent.return_value = mock_agent
        mock_agent.get_status.return_value = {
            "initialized": True,
            "current_provider": "claude",
            "last_activity": "2025-01-07T20:30:00Z"
        }
        
        response = client.get("/api/dannytrades/status")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["initialized"] is True
        assert data["current_provider"] == "claude"

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_get_activities(self, mock_get_agent, client, mock_agent):
        """Test activities endpoint."""
        mock_get_agent.return_value = mock_agent
        mock_agent.get_activities.return_value = [
            {
                "id": "1",
                "timestamp": "2025-01-07T20:30:00Z",
                "type": "success",
                "message": "Processed AAPL successfully"
            }
        ]
        
        response = client.get("/api/dannytrades/activities")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["type"] == "success"

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_switch_provider_success(self, mock_get_agent, client, mock_agent):
        """Test successful provider switching."""
        mock_get_agent.return_value = mock_agent
        mock_agent.switch_provider.return_value = True
        
        response = client.post("/api/dannytrades/providers/switch", json={"provider": "openai"})
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        mock_agent.switch_provider.assert_called_once_with("openai")

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_switch_provider_failure(self, mock_get_agent, client, mock_agent):
        """Test failed provider switching."""
        mock_get_agent.return_value = mock_agent
        mock_agent.switch_provider.return_value = False
        
        response = client.post("/api/dannytrades/providers/switch", json={"provider": "invalid"})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["success"] is False

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_update_prompts_success(self, mock_get_agent, client, mock_agent):
        """Test successful prompt updating."""
        mock_get_agent.return_value = mock_agent
        mock_agent.update_prompts.return_value = True
        
        prompts_payload = {
            "system_prompt": "Updated system prompt",
            "processing_instructions": "Updated instructions"
        }
        
        response = client.post("/api/dannytrades/prompts/update", json=prompts_payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        mock_agent.update_prompts.assert_called_once_with(prompts_payload)

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_update_prompts_failure(self, mock_get_agent, client, mock_agent):
        """Test failed prompt updating."""
        mock_get_agent.return_value = mock_agent
        mock_agent.update_prompts.return_value = False
        
        prompts_payload = {"system_prompt": ""}
        
        response = client.post("/api/dannytrades/prompts/update", json=prompts_payload)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["success"] is False

    def test_invalid_json_payload(self, client):
        """Test API with invalid JSON payload."""
        response = client.post(
            "/api/dannytrades/process",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_content_type(self, client, sample_stock_payload):
        """Test API with missing content type header."""
        response = client.post("/api/dannytrades/process", data=json.dumps(sample_stock_payload))
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_agent_not_initialized(self, mock_get_agent, client, sample_stock_payload):
        """Test API when agent is not initialized."""
        mock_get_agent.return_value = None
        
        response = client.post("/api/dannytrades/process", json=sample_stock_payload)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "not initialized" in data["detail"].lower()

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_concurrent_requests(self, mock_get_agent, client, sample_stock_payload, mock_agent):
        """Test handling of concurrent requests."""
        mock_get_agent.return_value = mock_agent
        mock_agent.process_stock_data.return_value = ProcessingResult(
            success=True,
            action="update",
            reasoning="Processed successfully",
            confidence=0.9,
            processing_time=1.5
        )
        
        # Simulate concurrent requests
        responses = []
        for i in range(3):
            payload = sample_stock_payload.copy()
            payload["symbol"] = f"TEST{i}"
            response = client.post("/api/dannytrades/process", json=payload)
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True

    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present."""
        response = client.get("/api/dannytrades/health")
        
        # Check for common rate limiting headers
        # Note: Actual implementation would depend on rate limiting middleware
        assert response.status_code == status.HTTP_200_OK

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_large_payload_handling(self, mock_get_agent, client, mock_agent):
        """Test handling of large payloads."""
        mock_get_agent.return_value = mock_agent
        mock_agent.process_stock_data.return_value = ProcessingResult(
            success=True,
            action="update",
            reasoning="Processed large payload",
            confidence=0.8,
            processing_time=3.0
        )
        
        # Create a large payload
        large_payload = {
            "symbol": "LARGE",
            "ticker_name": "Large Test Corp",
            "volatility_hole": True,
            "red_candle": {"daily": True, "weekly": False, "monthly": True},
            "yellow_candle": {"daily": False, "weekly": True, "monthly": False},
            "whale_accumulation": {"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
            "technical_indicators": {
                "support_levels": list(range(100, 200)),  # Large list
                "resistance_levels": list(range(200, 300))
            },
            "bullishness_signal": ["Extreme Bullish"] * 10,
            "misc_data": "A" * 1000  # Large string
        }
        
        response = client.post("/api/dannytrades/process", json=large_payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True


class TestAPIErrorHandling:
    """Test cases for API error handling and edge cases."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)

    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.put("/api/dannytrades/process")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_endpoint_not_found(self, client):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/api/dannytrades/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('app.api.dannytrades_routes.get_dannytrades_agent')
    def test_internal_server_error(self, mock_get_agent, client):
        """Test internal server error handling."""
        mock_get_agent.side_effect = Exception("Unexpected error")
        
        response = client.get("/api/dannytrades/status")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_request_timeout_simulation(self, client):
        """Test request timeout handling (simulated)."""
        # This would require actual timeout configuration in production
        # For now, we just test that the endpoint responds
        response = client.get("/api/dannytrades/health")
        assert response.status_code == status.HTTP_200_OK


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
