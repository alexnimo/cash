"""
Test suite for DannyTrades data models and validation.

Tests Pydantic models, data validation, serialization/deserialization,
and edge cases for the DannyTrades agent data structures.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
import json

from app.agents.dannytrades.models import (
    StockData, 
    ProcessingResult, 
    RedCandleData, 
    YellowCandleData,
    WhaleAccumulationData,
    TechnicalIndicators,
    AgentActivity,
    ProviderConfig
)


class TestStockDataModel:
    """Test cases for StockData model validation and serialization."""

    @pytest.fixture
    def valid_stock_data(self) -> Dict[str, Any]:
        """Valid stock data for testing."""
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

    def test_valid_stock_data_creation(self, valid_stock_data):
        """Test creation of StockData with valid data."""
        stock = StockData(**valid_stock_data)
        
        assert stock.symbol == "AAPL"
        assert stock.ticker_name == "Apple Inc."
        assert stock.volatility_hole is True
        assert stock.red_candle.daily is True
        assert stock.yellow_candle.weekly is True
        assert stock.whale_accumulation.daily == "Strong"
        assert len(stock.technical_indicators.support_levels) == 3
        assert "Extreme Bullish" in stock.bullishness_signal

    def test_required_fields_validation(self):
        """Test validation of required fields."""
        # Missing symbol
        with pytest.raises(ValueError, match="symbol"):
            StockData(
                volatility_hole=True,
                red_candle={"daily": True, "weekly": False, "monthly": True},
                yellow_candle={"daily": False, "weekly": True, "monthly": False},
                whale_accumulation={"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                technical_indicators={"support_levels": [145.0], "resistance_levels": [155.0]},
                bullishness_signal=["Hot"]
            )

    def test_symbol_validation(self):
        """Test symbol field validation."""
        # Empty symbol
        with pytest.raises(ValueError):
            StockData(
                symbol="",
                volatility_hole=True,
                red_candle={"daily": True, "weekly": False, "monthly": True},
                yellow_candle={"daily": False, "weekly": True, "monthly": False},
                whale_accumulation={"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                technical_indicators={"support_levels": [145.0], "resistance_levels": [155.0]},
                bullishness_signal=["Hot"]
            )

        # Symbol too long
        with pytest.raises(ValueError):
            StockData(
                symbol="A" * 20,  # Too long
                volatility_hole=True,
                red_candle={"daily": True, "weekly": False, "monthly": True},
                yellow_candle={"daily": False, "weekly": True, "monthly": False},
                whale_accumulation={"daily": "Strong", "weekly": "Moderate", "monthly": "Weak"},
                technical_indicators={"support_levels": [145.0], "resistance_levels": [155.0]},
                bullishness_signal=["Hot"]
            )

    def test_bullishness_signal_validation(self, valid_stock_data):
        """Test bullishness signal validation."""
        # Valid signals
        valid_signals = ["Hot", "Extreme Bullish", "Neutral", "Bearish"]
        for signal in valid_signals:
            data = valid_stock_data.copy()
            data["bullishness_signal"] = [signal]
            stock = StockData(**data)
            assert signal in stock.bullishness_signal

        # Invalid signal
        data = valid_stock_data.copy()
        data["bullishness_signal"] = ["Invalid Signal"]
        with pytest.raises(ValueError):
            StockData(**data)

    def test_whale_accumulation_validation(self, valid_stock_data):
        """Test whale accumulation data validation."""
        # Valid accumulation levels
        valid_levels = ["Strong", "Moderate", "Weak", "None"]
        for level in valid_levels:
            data = valid_stock_data.copy()
            data["whale_accumulation"] = {"daily": level, "weekly": level, "monthly": level}
            stock = StockData(**data)
            assert stock.whale_accumulation.daily == level

        # Invalid accumulation level
        data = valid_stock_data.copy()
        data["whale_accumulation"] = {"daily": "Invalid", "weekly": "Moderate", "monthly": "Weak"}
        with pytest.raises(ValueError):
            StockData(**data)

    def test_technical_indicators_validation(self, valid_stock_data):
        """Test technical indicators validation."""
        # Empty support levels
        data = valid_stock_data.copy()
        data["technical_indicators"] = {"support_levels": [], "resistance_levels": [155.0]}
        with pytest.raises(ValueError):
            StockData(**data)

        # Negative price levels
        data = valid_stock_data.copy()
        data["technical_indicators"] = {"support_levels": [-10.0], "resistance_levels": [155.0]}
        with pytest.raises(ValueError):
            StockData(**data)

    def test_optional_fields(self, valid_stock_data):
        """Test optional fields handling."""
        # Remove optional fields
        data = valid_stock_data.copy()
        del data["ticker_name"]
        del data["misc_data"]
        del data["patreon_link"]
        
        stock = StockData(**data)
        assert stock.ticker_name is None
        assert stock.misc_data is None
        assert stock.patreon_link is None

    def test_json_serialization(self, valid_stock_data):
        """Test JSON serialization and deserialization."""
        stock = StockData(**valid_stock_data)
        
        # Serialize to JSON
        json_data = stock.model_dump_json()
        assert isinstance(json_data, str)
        
        # Deserialize from JSON
        parsed_data = json.loads(json_data)
        reconstructed_stock = StockData(**parsed_data)
        
        assert reconstructed_stock.symbol == stock.symbol
        assert reconstructed_stock.volatility_hole == stock.volatility_hole

    def test_timestamp_auto_generation(self, valid_stock_data):
        """Test automatic timestamp generation."""
        stock = StockData(**valid_stock_data)
        
        assert stock.last_updated is not None
        assert isinstance(stock.last_updated, str)
        
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(stock.last_updated.replace('Z', '+00:00'))


class TestProcessingResultModel:
    """Test cases for ProcessingResult model."""

    def test_successful_result_creation(self):
        """Test creation of successful processing result."""
        result = ProcessingResult(
            success=True,
            action="update",
            reasoning="Strong bullish signals detected",
            confidence=0.95,
            processing_time=2.3,
            stock_symbol="AAPL"
        )
        
        assert result.success is True
        assert result.action == "update"
        assert result.confidence == 0.95
        assert result.processing_time == 2.3
        assert result.stock_symbol == "AAPL"

    def test_failed_result_creation(self):
        """Test creation of failed processing result."""
        result = ProcessingResult(
            success=False,
            error_message="LLM API error",
            processing_time=1.0
        )
        
        assert result.success is False
        assert result.error_message == "LLM API error"
        assert result.action is None
        assert result.reasoning is None

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        for confidence in [0.0, 0.5, 1.0]:
            result = ProcessingResult(
                success=True,
                action="update",
                confidence=confidence
            )
            assert result.confidence == confidence

        # Invalid confidence scores
        for invalid_confidence in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError):
                ProcessingResult(
                    success=True,
                    action="update",
                    confidence=invalid_confidence
                )

    def test_action_validation(self):
        """Test action field validation."""
        valid_actions = ["create", "update", "skip", "error"]
        
        for action in valid_actions:
            result = ProcessingResult(
                success=True,
                action=action
            )
            assert result.action == action

        # Invalid action
        with pytest.raises(ValueError):
            ProcessingResult(
                success=True,
                action="invalid_action"
            )

    def test_processing_time_validation(self):
        """Test processing time validation."""
        # Valid processing times
        result = ProcessingResult(success=True, processing_time=0.1)
        assert result.processing_time == 0.1

        # Negative processing time
        with pytest.raises(ValueError):
            ProcessingResult(success=True, processing_time=-1.0)


class TestNestedModels:
    """Test cases for nested data models."""

    def test_red_candle_data_model(self):
        """Test RedCandleData model."""
        red_candle = RedCandleData(daily=True, weekly=False, monthly=True)
        
        assert red_candle.daily is True
        assert red_candle.weekly is False
        assert red_candle.monthly is True

    def test_yellow_candle_data_model(self):
        """Test YellowCandleData model."""
        yellow_candle = YellowCandleData(daily=False, weekly=True, monthly=False)
        
        assert yellow_candle.daily is False
        assert yellow_candle.weekly is True
        assert yellow_candle.monthly is False

    def test_whale_accumulation_data_model(self):
        """Test WhaleAccumulationData model."""
        whale_data = WhaleAccumulationData(
            daily="Strong",
            weekly="Moderate", 
            monthly="Weak"
        )
        
        assert whale_data.daily == "Strong"
        assert whale_data.weekly == "Moderate"
        assert whale_data.monthly == "Weak"

        # Test invalid accumulation level
        with pytest.raises(ValueError):
            WhaleAccumulationData(
                daily="Invalid",
                weekly="Moderate",
                monthly="Weak"
            )

    def test_technical_indicators_model(self):
        """Test TechnicalIndicators model."""
        indicators = TechnicalIndicators(
            support_levels=[145.0, 140.0, 135.0],
            resistance_levels=[155.0, 160.0, 165.0]
        )
        
        assert len(indicators.support_levels) == 3
        assert len(indicators.resistance_levels) == 3
        assert indicators.support_levels[0] == 145.0

        # Test empty lists
        with pytest.raises(ValueError):
            TechnicalIndicators(support_levels=[], resistance_levels=[155.0])


class TestAgentActivityModel:
    """Test cases for AgentActivity model."""

    def test_activity_creation(self):
        """Test creation of agent activity."""
        activity = AgentActivity(
            type="processing",
            message="Processing AAPL data",
            details={"symbol": "AAPL", "action": "update"},
            duration=2.5
        )
        
        assert activity.type == "processing"
        assert activity.message == "Processing AAPL data"
        assert activity.details["symbol"] == "AAPL"
        assert activity.duration == 2.5
        assert activity.timestamp is not None

    def test_activity_type_validation(self):
        """Test activity type validation."""
        valid_types = ["processing", "success", "error", "warning", "info"]
        
        for activity_type in valid_types:
            activity = AgentActivity(
                type=activity_type,
                message="Test message"
            )
            assert activity.type == activity_type

        # Invalid type
        with pytest.raises(ValueError):
            AgentActivity(
                type="invalid_type",
                message="Test message"
            )

    def test_timestamp_auto_generation(self):
        """Test automatic timestamp generation."""
        activity = AgentActivity(
            type="info",
            message="Test message"
        )
        
        assert activity.timestamp is not None
        # Should be recent (within last minute)
        timestamp = datetime.fromisoformat(activity.timestamp.replace('Z', '+00:00'))
        now = datetime.now().replace(tzinfo=timestamp.tzinfo)
        diff = (now - timestamp).total_seconds()
        assert diff < 60  # Less than 1 minute old


class TestProviderConfigModel:
    """Test cases for ProviderConfig model."""

    def test_provider_config_creation(self):
        """Test creation of provider configuration."""
        config = ProviderConfig(
            name="claude",
            model_id="claude-3-sonnet-20240229",
            enabled=True,
            has_api_key=True,
            endpoint="https://api.anthropic.com"
        )
        
        assert config.name == "claude"
        assert config.model_id == "claude-3-sonnet-20240229"
        assert config.enabled is True
        assert config.has_api_key is True

    def test_provider_name_validation(self):
        """Test provider name validation."""
        valid_names = ["claude", "openai", "gemini", "ollama", "groq", "mistral"]
        
        for name in valid_names:
            config = ProviderConfig(
                name=name,
                model_id="test-model",
                enabled=True,
                has_api_key=False
            )
            assert config.name == name

        # Invalid provider name
        with pytest.raises(ValueError):
            ProviderConfig(
                name="invalid_provider",
                model_id="test-model",
                enabled=True,
                has_api_key=False
            )

    def test_optional_fields(self):
        """Test optional fields in provider config."""
        config = ProviderConfig(
            name="claude",
            model_id="claude-3-sonnet-20240229",
            enabled=True,
            has_api_key=True
        )
        
        assert config.endpoint is None
        assert config.current is False


class TestModelIntegration:
    """Test cases for model integration and complex scenarios."""

    def test_complete_stock_processing_flow(self):
        """Test complete flow from stock data to processing result."""
        # Create stock data
        stock_data = StockData(
            symbol="TSLA",
            volatility_hole=False,
            red_candle={"daily": False, "weekly": True, "monthly": False},
            yellow_candle={"daily": True, "weekly": False, "monthly": True},
            whale_accumulation={"daily": "Moderate", "weekly": "Strong", "monthly": "Strong"},
            technical_indicators={"support_levels": [240.0], "resistance_levels": [260.0]},
            bullishness_signal=["Hot"]
        )
        
        # Create processing result
        result = ProcessingResult(
            success=True,
            action="update",
            reasoning="Hot stock with strong weekly whale accumulation",
            confidence=0.85,
            processing_time=1.8,
            stock_symbol=stock_data.symbol
        )
        
        # Create activity log
        activity = AgentActivity(
            type="success",
            message=f"Successfully processed {stock_data.symbol}",
            details={
                "symbol": stock_data.symbol,
                "action": result.action,
                "confidence": result.confidence
            },
            duration=result.processing_time
        )
        
        assert stock_data.symbol == result.stock_symbol == "TSLA"
        assert result.success is True
        assert activity.type == "success"

    def test_error_scenario_models(self):
        """Test models in error scenarios."""
        # Failed processing result
        result = ProcessingResult(
            success=False,
            error_message="Invalid JSON structure in input data",
            processing_time=0.5,
            stock_symbol="INVALID"
        )
        
        # Error activity
        activity = AgentActivity(
            type="error",
            message="Failed to process stock data",
            details={
                "error": result.error_message,
                "symbol": result.stock_symbol
            }
        )
        
        assert result.success is False
        assert result.action is None
        assert activity.type == "error"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
