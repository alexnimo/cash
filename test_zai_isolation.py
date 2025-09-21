#!/usr/bin/env python3
"""
Test script to isolate Z.AI timeout issues.

This script tests:
1. Z.AI API call without MCP integration
2. MCP integration separately 
3. Combined Z.AI + MCP to identify bottleneck
"""

import asyncio
import os
import sys
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.dannytrades.simple_agent import SimpleDannyTradesAgent
from app.core.config_manager import ConfigManager

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('zai_isolation_test.log')
    ]
)
logger = logging.getLogger(__name__)

class ZAIIsolationTester:
    """Test suite to isolate Z.AI timeout issues."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        
    def create_test_stock_data(self) -> dict:
        """Create test stock data."""
        return {
            "symbol": "TIMEOUT_TEST",
            "price": 100.0,
            "change": 2.5,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
    
    async def test_zai_api_only(self) -> dict:
        """Test Z.AI API call without MCP integration."""
        logger.info("🧪 TEST 1: Z.AI API call without MCP integration")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Get Z.AI configuration
            config = self.config_manager.load_dannytrades_config()
            zai_config = config.get('providers', {}).get('zai', {})
            
            if not zai_config.get('api_key'):
                return {
                    "status": "skipped",
                    "reason": "No Z.AI API key configured"
                }
            
            api_key = zai_config.get('api_key')
            base_url = zai_config.get('endpoint', 'https://api.z.ai/api/coding/paas/v4')
            model = zai_config.get('model', 'glm-4.5')
            
            logger.info(f"🔧 Using endpoint: {base_url}")
            logger.info(f"🔧 Using model: {model}")
            
            # Create test prompt
            stock_data = self.create_test_stock_data()
            prompt = f"""
            Analyze this stock data:
            
            Symbol: {stock_data.get('symbol', 'N/A')}
            Price: ${stock_data.get('price', 'N/A')}
            Change: {stock_data.get('change', 'N/A')}%
            Volume: {stock_data.get('volume', 'N/A')}
            
            Provide investment analysis.
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a stock analysis expert. Provide concise investment analysis."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 1000
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": zai_config.get('user_agent', 'opencode/1.0'),
                "Accept-Language": zai_config.get('accept_language', 'en-US,en'),
                "X-Client-Type": "coding-agent",
                "X-Agent-Name": "opencode",
                "X-Agent-Version": "1.0"
            }
            
            logger.info("🌐 Sending request to Z.AI API...")
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                logger.info(f"📡 Z.AI API Response: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"❌ Z.AI API error {response.status_code}: {response.text}")
                    raise Exception(f"Z.AI API error {response.status_code}: {response.text}")
                
                result_data = response.json()
                content = result_data['choices'][0]['message']['content']
                
                processing_time = time.time() - start_time
                
                logger.info(f"✅ Z.AI API call completed in {processing_time:.2f}s")
                logger.info(f"📝 Response length: {len(content)} characters")
                
                return {
                    "status": "success",
                    "processing_time": processing_time,
                    "response_length": len(content),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Z.AI API test failed after {processing_time:.2f}s: {e}")
            return {
                "status": "error",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def test_mcp_integration_only(self) -> dict:
        """Test MCP integration with dummy analysis."""
        logger.info("🧪 TEST 2: MCP integration only (with dummy analysis)")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Check Notion API key
            notion_token = os.getenv("NOTION_API_KEY")
            if not notion_token:
                return {
                    "status": "skipped",
                    "reason": "No Notion API key found"
                }
            
            # Create agent instance
            agent = SimpleDannyTradesAgent()
            
            # Create dummy analysis
            dummy_analysis = "This is a test analysis for MCP integration testing. The stock shows positive momentum with good volume."
            
            # Test MCP integration
            stock_data = self.create_test_stock_data()
            
            logger.info("🔄 Starting MCP integration test...")
            
            result = await agent._unified_mcp_integration(stock_data, dummy_analysis)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ MCP integration test completed in {processing_time:.2f}s")
            
            return {
                "status": "success" if result.get('success') else "error",
                "processing_time": processing_time,
                "mcp_result": result
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ MCP integration test failed after {processing_time:.2f}s: {e}")
            return {
                "status": "error",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def test_combined_zai_mcp(self) -> dict:
        """Test combined Z.AI + MCP integration."""
        logger.info("🧪 TEST 3: Combined Z.AI + MCP integration")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Create agent instance
            agent = SimpleDannyTradesAgent()
            
            # Switch to Z.AI provider
            config = self.config_manager.load_dannytrades_config()
            zai_config = config.get('providers', {}).get('zai', {})
            
            if not zai_config.get('api_key'):
                return {
                    "status": "skipped",
                    "reason": "No Z.AI API key configured"
                }
            
            success = await agent.switch_provider(
                provider_name="zai",
                api_key=zai_config.get('api_key', ''),
                model=zai_config.get('model', 'glm-4.5'),
                endpoint=zai_config.get('endpoint', 'https://api.z.ai/api/coding/paas/v4')
            )
            
            if not success:
                return {
                    "status": "error",
                    "reason": "Failed to switch to Z.AI provider"
                }
            
            # Test full processing
            stock_data = self.create_test_stock_data()
            
            logger.info("🚀 Starting combined Z.AI + MCP test...")
            
            result = await agent.process_stock_data(stock_data)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Combined test completed in {processing_time:.2f}s")
            
            return {
                "status": "success" if result.success else "error",
                "processing_time": processing_time,
                "result": {
                    "success": result.success,
                    "error_message": result.error_message if not result.success else None,
                    "reasoning_length": len(result.reasoning) if result.reasoning else 0
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Combined test failed after {processing_time:.2f}s: {e}")
            return {
                "status": "error",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def run_all_tests(self) -> dict:
        """Run all isolation tests."""
        logger.info("🚀 Starting Z.AI Isolation Test Suite")
        logger.info("=" * 80)
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test 1: Z.AI API only
        results["tests"]["zai_api_only"] = await self.test_zai_api_only()
        
        # Test 2: MCP integration only
        results["tests"]["mcp_integration_only"] = await self.test_mcp_integration_only()
        
        # Test 3: Combined Z.AI + MCP
        results["tests"]["combined_zai_mcp"] = await self.test_combined_zai_mcp()
        
        return results
    
    def print_results(self, results: dict):
        """Print test results."""
        print("\n" + "="*80)
        print("🧪 Z.AI ISOLATION TEST RESULTS")
        print("="*80)
        
        for test_name, test_result in results.get("tests", {}).items():
            status_icon = {
                "success": "✅",
                "error": "❌", 
                "skipped": "⏭️"
            }.get(test_result.get("status"), "❓")
            
            print(f"\n{status_icon} {test_name.upper().replace('_', ' ')}")
            print(f"   Status: {test_result.get('status', 'unknown')}")
            
            if "processing_time" in test_result:
                print(f"   Processing Time: {test_result['processing_time']:.2f}s")
            
            if test_result.get("status") == "error":
                print(f"   Error: {test_result.get('error', 'Unknown error')}")
            elif test_result.get("status") == "skipped":
                print(f"   Reason: {test_result.get('reason', 'Unknown reason')}")
            elif test_result.get("status") == "success":
                if "response_length" in test_result:
                    print(f"   Response Length: {test_result['response_length']} chars")
                if "mcp_result" in test_result:
                    mcp_success = test_result['mcp_result'].get('success', False)
                    print(f"   MCP Success: {mcp_success}")

async def main():
    """Main test function."""
    tester = ZAIIsolationTester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        # Print results
        tester.print_results(results)
        
        # Save results to file
        import json
        results_file = "zai_isolation_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"📋 Detailed logs saved to: zai_isolation_test.log")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
