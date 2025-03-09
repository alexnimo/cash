import pytest
from unittest.mock import Mock, patch
from app.tools.notion_tool_v2 import NotionTool, NotionAdvancedToolSpec
from datetime import datetime

@pytest.fixture
def mock_notion():
    with patch('app.tools.notion_tool_v2.Client') as mock:
        yield mock

@pytest.fixture
def notion_tool(mock_notion):
    with patch.dict('os.environ', {
        'NOTION_API_KEY': 'test-key',
        'NOTION_DATABASE_ID': 'test-db'
    }):
        return NotionTool()

def test_create_technical_analysis_section(notion_tool, mock_notion):
    mock_blocks = Mock()
    mock_notion.return_value.blocks.children.append = mock_blocks
    
    content = {"RSI": "70", "MACD": "Bullish"}
    result = notion_tool.create_technical_analysis_section("test-page", content)
    
    mock_blocks.assert_called_once()
    call_args = mock_blocks.call_args[1]
    assert "children" in call_args
    assert len(call_args["children"]) > 1

def test_upload_chart_image(notion_tool, mock_notion):
    mock_blocks = Mock()
    mock_notion.return_value.blocks.children.append = mock_blocks
    
    result = notion_tool.upload_chart_image("test-page", "http://test.com/image.png")
    
    mock_blocks.assert_called_once()
    call_args = mock_blocks.call_args[1]
    assert "children" in call_args
    assert call_args["children"][0]["type"] == "image"

def test_create_stock_page(notion_tool, mock_notion):
    mock_pages = Mock()
    mock_notion.return_value.pages.create = mock_pages
    mock_pages.return_value = {"id": "test-id"}
    
    content = {
        "technical_analysis": {"RSI": "70"},
        "market_insights": {"trend": "bullish"},
        "chart_url": "http://test.com/chart.png"
    }
    
    result = notion_tool.create_stock_page("AAPL", content)
    
    mock_pages.assert_called_once()
    assert result == "test-id"

def test_create_or_update_stock_page_existing(notion_tool, mock_notion):
    mock_query = Mock()
    mock_notion.return_value.databases.query = mock_query
    mock_query.return_value = {
        "results": [{"id": "existing-page"}]
    }
    
    content = {"technical_analysis": {"RSI": "70"}}
    result = notion_tool.create_or_update_stock_page("AAPL", content)
    
    assert result == "existing-page"
