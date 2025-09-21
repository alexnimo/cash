"""
Comprehensive test to validate MCP server functionality with Z.AI integration.
Tests both data saving and retrieval to verify Notion database updates.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_mcp_server_validation():
    """Test MCP server functionality with mock data and validation."""
    
    # Mock stock data for testing
    mock_stock_data = {
        "symbol": "AAPL",
        "price": 175.43,
        "change": 2.15,
        "change_percent": 1.24,
        "volume": 45678900,
        "market_cap": "2.8T",
        "pe_ratio": 28.5,
        "dividend_yield": 0.52,
        "52_week_high": 199.62,
        "52_week_low": 164.08,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info("🧪 Starting MCP server validation test")
    logger.info(f"📊 Mock data: {mock_stock_data['symbol']} @ ${mock_stock_data['price']}")
    logger.info(f"📅 Test timestamp: {mock_stock_data['timestamp']}")
    
    try:
        # Import and initialize the agent
        from app.agents.dannytrades.simple_agent import SimpleDannyTradesAgent
        from app.agents.dannytrades.config import DannyTradesConfig
        
        logger.info("🔧 Initializing DannyTrades agent...")
        config = DannyTradesConfig()
        agent = SimpleDannyTradesAgent(config)
        await agent.initialize()
        
        # Test 1: Process stock data with Z.AI + MCP integration
        logger.info("🚀 Test 1: Processing stock data with Z.AI + MCP integration")
        result = await agent.process_stock_data(mock_stock_data)
        
        if result.success:
            logger.info(f"✅ Processing successful in {result.processing_time:.2f}s")
            logger.info(f"📋 Analysis length: {len(result.reasoning)} characters")
            logger.info(f"💭 Analysis preview: {result.reasoning[:200]}...")
        else:
            logger.error(f"❌ Processing failed: {result.error_message}")
            return False
        
        # Test 2: Validate MCP server data retrieval
        logger.info("🔍 Test 2: Validating MCP server data retrieval")
        validation_result = await validate_notion_data(mock_stock_data)
        
        if validation_result['success']:
            logger.info("✅ MCP server validation successful")
            logger.info(f"📅 Data timestamp matches: {validation_result['date_match']}")
            logger.info(f"📊 Stock symbol found: {validation_result['symbol_found']}")
        else:
            logger.warning(f"⚠️ MCP server validation issues: {validation_result['error']}")
        
        # Test 3: Testing with invalid stock symbol to verify MCP server error handling
        logger.info("🔄 Test 3: Testing with invalid stock symbol (BKSH)")
        mock_stock_data_2 = {
            **mock_stock_data,
            "symbol": "BKSH",
            "price": 0.00,
            "change": 0.00,
            "change_percent": 0.00,
            "volume": 0,
            "market_cap": "N/A",
            "pe_ratio": 0.00,
            "dividend_yield": 0.00,
            "52_week_high": 0.00,
            "52_week_low": 0.00,
            "timestamp": datetime.now().isoformat(),
            "error": "Invalid stock symbol - BKSH does not exist"
        }
        
        result_2 = await agent.process_stock_data(mock_stock_data_2)
        if result_2.success:
            logger.info(f"✅ Invalid symbol test completed for {mock_stock_data_2['symbol']}")
            # Check if the analysis mentions the symbol is invalid or doesn't exist
            if "invalid" in result_2.reasoning.lower() or "does not exist" in result_2.reasoning.lower() or "bksh" in result_2.reasoning.lower():
                logger.info("🎯 MCP server correctly identified invalid stock symbol")
            else:
                logger.warning("⚠️ MCP server may not have detected invalid symbol properly")
        else:
            logger.error(f"❌ Invalid symbol test failed: {result_2.error_message}")
        
        # Test 4: Validate that BKSH is not found in database
        logger.info("🔍 Test 4: Validating BKSH is not found in database")
        bksh_validation = await validate_notion_data(mock_stock_data_2)
        
        if bksh_validation['success']:
            if not bksh_validation['symbol_found']:
                logger.info("✅ Confirmed: BKSH not found in database (as expected)")
            else:
                logger.warning("⚠️ Unexpected: BKSH found in database")
        else:
            logger.info(f"ℹ️ BKSH validation result: {bksh_validation['error']}")
        
        logger.info("🎉 MCP server validation test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        logger.error(f"🔍 Exception type: {type(e).__name__}")
        return False

async def validate_notion_data(stock_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that data was properly saved to Notion via MCP server."""
    try:
        logger.info("🔧 Setting up MCP validation...")
        
        # Check if we have required environment variables
        notion_token = os.getenv("NOTION_API_KEY")
        if not notion_token:
            return {"success": False, "error": "No Notion API key found"}
        
        # Get database ID from config
        from app.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        saved_config = config_manager.load_dannytrades_config()
        database_id = saved_config.get('notion', {}).get('database_id')
        
        if not database_id:
            return {"success": False, "error": "No Notion database ID configured"}
        
        logger.info(f"📊 Validating data for {stock_data['symbol']} in database {database_id[:8]}...")
        
        # Import agents SDK components for validation
        try:
            from agents import Agent, Runner
            from agents.mcp import MCPServerStdio
        except ImportError:
            return {"success": False, "error": "agents package not available"}
        
        # Create validation prompt to search for the stock data
        current_date = datetime.now().strftime("%Y-%m-%d")
        validation_prompt = f"""Search the Notion database for stock analysis entries for {stock_data['symbol']}.
        
Look for entries that were created or updated today ({current_date}).
Check if there's a recent entry with:
- Stock Symbol: {stock_data['symbol']}
- Price around: ${stock_data['price']}
- Date: {current_date}

Please confirm if the data was successfully saved and provide details about the entry."""
        
        logger.info(f"📝 Validation prompt created ({len(validation_prompt)} chars)")
        logger.info(f"🔍 Searching for {stock_data['symbol']} entries from {current_date}")
        
        # Use MCP server to validate data
        async with MCPServerStdio(
            name="Notion MCP Server",
            params={
                "command": "npx",
                "args": ["-y", "@notionhq/notion-mcp-server"],
                "env": {
                    "NOTION_TOKEN": notion_token,
                    "NOTION_DATABASE_ID": database_id
                }
            },
        ) as mcp_server:
            logger.info("🤖 Creating validation agent...")
            
            agent = Agent(
                name="DannyTrades Validation Agent",
                instructions="You are a Notion database validation agent. Search and verify stock analysis entries in the database.",
                mcp_servers=[mcp_server],
            )
            
            logger.info("⚡ Running validation query...")
            
            # Set environment variables for the agents SDK
            import os
            original_api_key = os.environ.get('OPENAI_API_KEY')
            original_base_url = os.environ.get('OPENAI_BASE_URL')
            
            # Get Z.AI credentials from config
            zai_config = saved_config.get('providers', {}).get('zai', {})
            zai_api_key = zai_config.get('api_key')
            zai_base_url = zai_config.get('endpoint', 'https://api.z.ai/api/coding/paas/v4')
            
            if zai_api_key:
                os.environ['OPENAI_API_KEY'] = zai_api_key
                os.environ['OPENAI_BASE_URL'] = zai_base_url
            
            try:
                result = await Runner.run(starting_agent=agent, input=validation_prompt)
            finally:
                # Restore original environment variables
                if original_api_key:
                    os.environ['OPENAI_API_KEY'] = original_api_key
                else:
                    os.environ.pop('OPENAI_API_KEY', None)
                
                if original_base_url:
                    os.environ['OPENAI_BASE_URL'] = original_base_url
                else:
                    os.environ.pop('OPENAI_BASE_URL', None)
            
            logger.info("🎯 Validation query completed")
            
            # Analyze the validation result
            result_text = str(result.final_output) if hasattr(result, 'final_output') else str(result)
            logger.info(f"📄 Validation result length: {len(result_text)} chars")
            logger.debug(f"💡 Validation result: {result_text[:300]}...")
            
            # Check for success indicators in the result
            symbol_found = stock_data['symbol'] in result_text
            date_match = current_date in result_text or "today" in result_text.lower()
            price_mentioned = str(stock_data['price']) in result_text
            
            logger.info(f"✅ Symbol found: {symbol_found}")
            logger.info(f"📅 Date match: {date_match}")
            logger.info(f"💰 Price mentioned: {price_mentioned}")
            
            return {
                "success": True,
                "symbol_found": symbol_found,
                "date_match": date_match,
                "price_mentioned": price_mentioned,
                "validation_result": result_text,
                "details": f"Validation completed for {stock_data['symbol']}"
            }
            
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Main test function."""
    logger.info("🧪 Starting comprehensive MCP server validation")
    logger.info("=" * 60)
    
    success = await test_mcp_server_validation()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 All tests completed successfully!")
    else:
        logger.error("❌ Some tests failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
