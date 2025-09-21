"""
Notion MCP server integration for DannyTrades agent.

This module provides MCP server functionality for interacting with Notion databases
specifically for the DannyTrades stock data processing system.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from notion_client import Client
from notion_client.errors import APIResponseError
import logging

logger = logging.getLogger(__name__)


class NotionMCPServer:
    """
    MCP server for Notion database operations.
    
    This class provides the MCP server functionality for interacting with
    Notion databases, specifically designed for stock data management.
    """
    
    def __init__(self, api_key: Optional[str] = None, database_id: Optional[str] = None):
        """
        Initialize the Notion MCP server.
        
        Args:
            api_key: Notion API key (defaults to environment variable)
            database_id: Notion database ID (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID", "20ef056c191c80068fd5f728fee3bda0")
        
        if not self.api_key:
            raise ValueError("Notion API key is required")
        
        self.client = Client(auth=self.api_key)
    
    async def query_database(self, filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the Notion database with filter criteria.
        
        Args:
            filter_criteria: Notion API filter criteria
            
        Returns:
            List of matching database entries
        """
        try:
            response = self.client.databases.query(
                database_id=self.database_id,
                filter=filter_criteria
            )
            return response.get("results", [])
        except APIResponseError as e:
            logger.error(f"Notion API error in query_database: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in query_database: {e}")
            raise
    
    async def create_page(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new page in the Notion database.
        
        Args:
            properties: Page properties following Notion API format
            
        Returns:
            Created page data
        """
        try:
            response = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties
            )
            return response
        except APIResponseError as e:
            logger.error(f"Notion API error in create_page: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in create_page: {e}")
            raise
    
    async def update_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing page in the Notion database.
        
        Args:
            page_id: ID of the page to update
            properties: Updated properties following Notion API format
            
        Returns:
            Updated page data
        """
        try:
            response = self.client.pages.update(
                page_id=page_id,
                properties=properties
            )
            return response
        except APIResponseError as e:
            logger.error(f"Notion API error in update_page: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in update_page: {e}")
            raise
    
    async def find_stock_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Find a stock entry by symbol.
        
        Args:
            symbol: Stock symbol to search for
            
        Returns:
            Stock page data if found, None otherwise
        """
        filter_criteria = {
            "property": "Stock Ticker",
            "title": {
                "equals": symbol
            }
        }
        
        results = await self.query_database(filter_criteria)
        return results[0] if results else None
    

    def format_stock_properties(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format stock data into Notion properties format.
        
        Args:
            stock_data: Stock data dictionary
            
        Returns:
            Formatted properties for Notion API
        """
        properties = {}
        
        # Stock Ticker (Title)
        if "symbol" in stock_data:
            properties["Stock Ticker"] = {
                "title": [
                    {"text": {"content": stock_data["symbol"]}}
                ]
            }
        
        # Red/Yellow Candle (Multi-select)
        candle_options = []
        if "red_candle" in stock_data and stock_data["red_candle"]:
            red_candle = stock_data["red_candle"]
            if red_candle.get("daily"):
                candle_options.append({"name": "Red-Daily"})
            if red_candle.get("weekly"):
                candle_options.append({"name": "Red-Weekly"})
            if red_candle.get("monthly"):
                candle_options.append({"name": "Red-Monthly"})
        
        if "yellow_candle" in stock_data and stock_data["yellow_candle"]:
            yellow_candle = stock_data["yellow_candle"]
            if yellow_candle.get("daily"):
                candle_options.append({"name": "Yellow-Daily"})
            if yellow_candle.get("weekly"):
                candle_options.append({"name": "Yellow-Weekly"})
            if yellow_candle.get("monthly"):
                candle_options.append({"name": "Yellow-Monthly"})
        
        if candle_options:
            properties["Candle"] = {"multi_select": candle_options}
        
        # Volatility Hole (Checkbox)
        if "volatility_hole" in stock_data and stock_data["volatility_hole"]:
            properties["Volatility Hole"] = {
                "checkbox": str(stock_data["volatility_hole"]).lower() in ["yes", "true", "1"]
            }
        
        # Whale Accumulation (Rich Text)
        if "whale_accumulation" in stock_data and stock_data["whale_accumulation"]:
            whale_acc = stock_data["whale_accumulation"]
            whale_text = []
            if whale_acc.get("daily"):
                whale_text.append(f"Daily: {whale_acc['daily']}")
            if whale_acc.get("weekly"):
                whale_text.append(f"Weekly: {whale_acc['weekly']}")
            if whale_acc.get("monthly"):
                whale_text.append(f"Monthly: {whale_acc['monthly']}")
            
            if whale_text:
                properties["Whale Accumulation"] = {
                    "rich_text": [
                        {"text": {"content": " | ".join(whale_text)}}
                    ]
                }
        
        # Notes (Rich Text) - using misc_data
        if "misc_data" in stock_data and stock_data["misc_data"]:
            properties["Notes"] = {
                "rich_text": [
                    {"text": {"content": stock_data["misc_data"]}}
                ]
            }
        
        # Tags (Multi-select) - using bullishness_signal
        if "bullishness_signal" in stock_data and stock_data["bullishness_signal"]:
            bullish_options = []
            for signal in stock_data["bullishness_signal"]:
                bullish_options.append({"name": signal})
            properties["Tags"] = {"multi_select": bullish_options}
        
        # Technical (Rich Text) - using technical_indicators
        if "technical_indicators" in stock_data and stock_data["technical_indicators"]:
            tech_indicators = stock_data["technical_indicators"]
            support_resistance_text = []
            
            if tech_indicators.get("support_levels"):
                support_text = ", ".join(map(str, tech_indicators["support_levels"]))
                support_resistance_text.append(f"Support Levels: {support_text}")
            
            if tech_indicators.get("resistance_levels"):
                resistance_text = ", ".join(map(str, tech_indicators["resistance_levels"]))
                support_resistance_text.append(f"Resistance Levels: {resistance_text}")
            
            if support_resistance_text:
                properties["Technical"] = {
                    "rich_text": [
                        {"text": {"content": " | ".join(support_resistance_text)}}
                    ]
                }
        
        # URL (URL) - using patreon_link
        if "patreon_link" in stock_data and stock_data["patreon_link"]:
            properties["URL"] = {"url": stock_data["patreon_link"]}
        
        return properties
    
    async def process_stock_update(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a stock update (create or update existing entry).
        
        Args:
            stock_data: Stock data to process
            
        Returns:
            Processing result with page information
        """
        try:
            symbol = stock_data.get("symbol")
            if not symbol:
                raise ValueError("Stock symbol is required")
            
            # Check if stock already exists
            existing_page = await self.find_stock_by_symbol(symbol)
            
            # Format properties
            properties = self.format_stock_properties(stock_data)
            
            if existing_page:
                # Update existing page
                page_id = existing_page["id"]
                updated_page = await self.update_page(page_id, properties)
                
                return {
                    "action": "updated",
                    "symbol": symbol,
                    "page_id": page_id,
                    "page_data": updated_page
                }
            else:
                # Create new page
                new_page = await self.create_page(properties)
                
                return {
                    "action": "created",
                    "symbol": symbol,
                    "page_id": new_page["id"],
                    "page_data": new_page
                }
                
        except Exception as e:
            logger.error(f"Error processing stock update for {stock_data.get('symbol', 'unknown')}: {e}")
            raise


# MCP Server Tools for Agno integration
class NotionMCPTools:
    """
    MCP tools for Agno agent integration.
    
    This class provides the tools that can be used by the Agno agent
    to interact with the Notion database.
    """
    
    def __init__(self, notion_server: NotionMCPServer):
        """
        Initialize MCP tools.
        
        Args:
            notion_server: Notion MCP server instance
        """
        self.notion_server = notion_server
    
    async def query_stock_database(self, symbol: str) -> Dict[str, Any]:
        """
        Query stock database for a specific symbol.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            Stock data if found
        """
        result = await self.notion_server.find_stock_by_symbol(symbol)
        return result or {"found": False, "symbol": symbol}
    
    async def update_stock_entry(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update or create a stock entry.
        
        Args:
            stock_data: Stock data to update
            
        Returns:
            Update result
        """
        return await self.notion_server.process_stock_update(stock_data)
    
    async def process_multiple_stocks(self, stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple stock entries.
        
        Args:
            stocks_data: List of stock data dictionaries
            
        Returns:
            List of update results
        """
        results = []
        
        for stock_data in stocks_data:
            try:
                result = await self.update_stock_entry(stock_data)
                results.append(result)
            except Exception as e:
                results.append({
                    "action": "failed",
                    "symbol": stock_data.get("symbol", "unknown"),
                    "error": str(e)
                })
        
        return results
    
    async def cleanup(self):
        """
        Cleanup resources (placeholder for future use).
        """
        pass
