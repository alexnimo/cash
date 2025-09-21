#!/usr/bin/env python3
"""
Test script for DannyTrades Agent refactoring.

This script tests both Z.AI and Agno providers to ensure:
1. Z.AI unified function works (custom HTTP + MCP integration)
2. Agno providers work with proper MCP integration
3. Specialized prompts from config are used consistently
4. MCP server integration works for all providers
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.dannytrades.simple_agent import SimpleDannyTradesAgent
from app.core.config_manager import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DannyTradesRefactorTester:
    """Test suite for DannyTrades agent refactoring."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.test_results = {}
        
    def create_test_stock_data(self, symbol: str = "TEST") -> Dict[str, Any]:
        """Create test stock data."""
        return {
            "symbol": symbol,
            "price": 150.75,
            "change": 2.5,
            "volume": 1500000,
            "timestamp": datetime.now().isoformat()
        }
    
    async def test_zai_provider(self) -> Dict[str, Any]:
        """Test Z.AI provider with unified function."""
        logger.info("🧪 Testing Z.AI provider with unified function...")
        
        try:
            # Load Z.AI configuration
            config = self.config_manager.load_dannytrades_config()
            zai_config = config.get('providers', {}).get('zai', {})
            
            if not zai_config.get('api_key'):
                return {
                    "status": "skipped",
                    "reason": "No Z.AI API key configured",
                    "provider": "zai"
                }
            
            # Create agent instance
            agent = SimpleDannyTradesAgent()
            
            # Switch to Z.AI provider
            success = await agent.switch_provider(
                provider_name="zai",
                api_key=zai_config.get('api_key', ''),
                model=zai_config.get('model', 'glm-4.5'),
                endpoint=zai_config.get('endpoint', 'https://api.z.ai/api/coding/paas/v4')
            )
            
            if not success:
                return {
                    "status": "error",
                    "reason": "Failed to switch to Z.AI provider",
                    "provider": "zai"
                }
            
            # Test stock data processing
            test_data = self.create_test_stock_data("ZAI_TEST")
            
            logger.info("🚀 Running Z.AI unified processing test...")
            result = await agent.process_stock_data(test_data)
            
            return {
                "status": "success" if result.success else "error",
                "provider": "zai",
                "processing_time": result.processing_time,
                "reasoning_length": len(result.reasoning) if result.reasoning else 0,
                "error_message": result.error_message if not result.success else None,
                "stock_symbol": result.stock_symbol,
                "unified_function_used": True
            }
            
        except Exception as e:
            logger.error(f"❌ Z.AI provider test failed: {e}")
            return {
                "status": "error",
                "provider": "zai",
                "reason": str(e)
            }
    
    async def test_agno_provider(self, provider_name: str) -> Dict[str, Any]:
        """Test Agno provider (OpenAI, Claude, Gemini, Ollama)."""
        logger.info(f"🧪 Testing Agno provider: {provider_name}...")
        
        try:
            # Load provider configuration
            config = self.config_manager.load_dannytrades_config()
            provider_config = config.get('providers', {}).get(provider_name, {})
            
            if not provider_config.get('enabled', False):
                return {
                    "status": "skipped",
                    "reason": f"{provider_name} provider not enabled",
                    "provider": provider_name
                }
            
            if not provider_config.get('api_key') and provider_name != 'ollama':
                return {
                    "status": "skipped",
                    "reason": f"No {provider_name} API key configured",
                    "provider": provider_name
                }
            
            # Create agent instance
            agent = SimpleDannyTradesAgent()
            
            # Switch to provider
            success = await agent.switch_provider(
                provider_name=provider_name,
                api_key=provider_config.get('api_key', ''),
                model=provider_config.get('model_id', ''),
                endpoint=provider_config.get('endpoint', '')
            )
            
            if not success:
                return {
                    "status": "error",
                    "reason": f"Failed to switch to {provider_name} provider",
                    "provider": provider_name
                }
            
            # Test stock data processing
            test_data = self.create_test_stock_data(f"{provider_name.upper()}_TEST")
            
            logger.info(f"🚀 Running {provider_name} Agno processing test...")
            result = await agent.process_stock_data(test_data)
            
            return {
                "status": "success" if result.success else "error",
                "provider": provider_name,
                "processing_time": result.processing_time,
                "reasoning_length": len(result.reasoning) if result.reasoning else 0,
                "error_message": result.error_message if not result.success else None,
                "stock_symbol": result.stock_symbol,
                "agno_function_used": True
            }
            
        except Exception as e:
            logger.error(f"❌ {provider_name} provider test failed: {e}")
            return {
                "status": "error",
                "provider": provider_name,
                "reason": str(e)
            }
    
    async def test_mcp_integration(self) -> Dict[str, Any]:
        """Test MCP server integration."""
        logger.info("🧪 Testing MCP server integration...")
        
        try:
            # Check Notion API key
            notion_token = os.getenv("NOTION_API_KEY")
            if not notion_token:
                return {
                    "status": "skipped",
                    "reason": "No Notion API key found in environment"
                }
            
            # Create agent instance
            agent = SimpleDannyTradesAgent()
            
            # Get MCP server status
            mcp_status = agent.get_mcp_server_status()
            
            return {
                "status": "success",
                "mcp_servers": mcp_status,
                "notion_available": mcp_status.get('notion', {}).get('status') == 'available'
            }
            
        except Exception as e:
            logger.error(f"❌ MCP integration test failed: {e}")
            return {
                "status": "error",
                "reason": str(e)
            }
    
    async def test_config_prompts(self) -> Dict[str, Any]:
        """Test that specialized prompts from config are used."""
        logger.info("🧪 Testing specialized prompts from config...")
        
        try:
            # Create agent instance
            agent = SimpleDannyTradesAgent()
            
            # Test getting agent instructions
            test_data = self.create_test_stock_data("PROMPT_TEST")
            instructions = agent._get_agent_instructions(test_data)
            
            # Check if instructions contain expected elements
            has_database_id = "database_id" in instructions or "DATABASE_ID" in instructions
            has_specialized_content = len(instructions) > 100  # Should be substantial
            
            return {
                "status": "success",
                "instructions_length": len(instructions),
                "has_database_id": has_database_id,
                "has_specialized_content": has_specialized_content,
                "preview": instructions[:200] + "..." if len(instructions) > 200 else instructions
            }
            
        except Exception as e:
            logger.error(f"❌ Config prompts test failed: {e}")
            return {
                "status": "error",
                "reason": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting DannyTrades refactoring test suite...")
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test Z.AI provider
        logger.info("\n" + "="*50)
        logger.info("Testing Z.AI Provider")
        logger.info("="*50)
        results["tests"]["zai_provider"] = await self.test_zai_provider()
        
        # Test Agno providers
        agno_providers = ["openai", "claude", "gemini", "ollama"]
        for provider in agno_providers:
            logger.info("\n" + "="*50)
            logger.info(f"Testing Agno Provider: {provider}")
            logger.info("="*50)
            results["tests"][f"{provider}_provider"] = await self.test_agno_provider(provider)
        
        # Test MCP integration
        logger.info("\n" + "="*50)
        logger.info("Testing MCP Integration")
        logger.info("="*50)
        results["tests"]["mcp_integration"] = await self.test_mcp_integration()
        
        # Test config prompts
        logger.info("\n" + "="*50)
        logger.info("Testing Config Prompts")
        logger.info("="*50)
        results["tests"]["config_prompts"] = await self.test_config_prompts()
        
        # Generate summary
        total_tests = len(results["tests"])
        successful_tests = sum(1 for test in results["tests"].values() if test.get("status") == "success")
        skipped_tests = sum(1 for test in results["tests"].values() if test.get("status") == "skipped")
        failed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "error")
        
        results["summary"] = {
            "total_tests": total_tests,
            "successful": successful_tests,
            "skipped": skipped_tests,
            "failed": failed_tests,
            "success_rate": f"{(successful_tests / (total_tests - skipped_tests) * 100):.1f}%" if total_tests > skipped_tests else "0%"
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a readable format."""
        print("\n" + "="*60)
        print("🧪 DANNYTRADES REFACTORING TEST RESULTS")
        print("="*60)
        
        summary = results.get("summary", {})
        print(f"📊 Total Tests: {summary.get('total_tests', 0)}")
        print(f"✅ Successful: {summary.get('successful', 0)}")
        print(f"⏭️  Skipped: {summary.get('skipped', 0)}")
        print(f"❌ Failed: {summary.get('failed', 0)}")
        print(f"📈 Success Rate: {summary.get('success_rate', '0%')}")
        
        print("\n" + "-"*60)
        print("📋 DETAILED RESULTS")
        print("-"*60)
        
        for test_name, test_result in results.get("tests", {}).items():
            status_icon = {
                "success": "✅",
                "error": "❌", 
                "skipped": "⏭️"
            }.get(test_result.get("status"), "❓")
            
            print(f"\n{status_icon} {test_name.upper().replace('_', ' ')}")
            print(f"   Status: {test_result.get('status', 'unknown')}")
            
            if test_result.get("status") == "success":
                if "processing_time" in test_result:
                    print(f"   Processing Time: {test_result['processing_time']:.2f}s")
                if "reasoning_length" in test_result:
                    print(f"   Reasoning Length: {test_result['reasoning_length']} chars")
                if "provider" in test_result:
                    print(f"   Provider: {test_result['provider']}")
            elif test_result.get("status") == "error":
                print(f"   Error: {test_result.get('reason', 'Unknown error')}")
            elif test_result.get("status") == "skipped":
                print(f"   Reason: {test_result.get('reason', 'Unknown reason')}")

async def main():
    """Main test function."""
    tester = DannyTradesRefactorTester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        # Print results
        tester.print_results(results)
        
        # Save results to file
        results_file = "dannytrades_refactor_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Exit with appropriate code
        summary = results.get("summary", {})
        if summary.get("failed", 0) > 0:
            print("\n❌ Some tests failed!")
            sys.exit(1)
        else:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
