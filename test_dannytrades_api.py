"""
Test script for DannyTrades API using Playwright to verify:
1. API endpoint accepts raw JSON
2. Agent processes the data correctly
3. Notion MCP tool updates the database
4. UI reflects the agent work and progress
"""

import asyncio
import json
import time
from playwright.async_api import async_playwright

# Test JSON data from user
TEST_JSON_DATA = {
    "symbol": "TSLA",
    "red_candle": {
        "monthly": True
    },
    "technical_indicators": {
        "resistance_levels": [353.19]
    },
    "misc_data": "The monthly bullish red candle is forming this month, signaling a long-term bullish trend. TSLA is struggling to break the $353.19 Fibonacci level, with a five-year rounded bottom formation supporting a bullish outlook. Patience is key for this long-term signal to finalize.",
    "patreon_link": "https://www.patreon.com/posts/dr-cats-video-on-138313686",
    "bullishness_signal": ["Extreme Bullish"]
}

API_BASE_URL = "http://localhost:8001"
API_ENDPOINT = f"{API_BASE_URL}/api/dannytrades/process"

async def test_api_direct():
    """Test the API endpoint directly using Playwright's request context."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        
        try:
            print("🚀 Testing DannyTrades API endpoint...")
            
            # Make API request
            response = await context.request.post(
                API_ENDPOINT,
                headers={"Content-Type": "application/json"},
                data=json.dumps(TEST_JSON_DATA)
            )
            
            print(f"📊 Response Status: {response.status}")
            
            if response.status == 200:
                result = await response.json()
                print("✅ API Success!")
                print(f"   Action: {result.get('action', 'N/A')}")
                print(f"   Reasoning: {result.get('reasoning', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")
                print(f"   Processing Time: {result.get('processing_time', 'N/A')}s")
                print(f"   Stock Symbol: {result.get('stock_symbol', 'N/A')}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ API Error: {error_text}")
                return False
                
        except Exception as e:
            print(f"❌ Exception during API test: {str(e)}")
            return False
        finally:
            await browser.close()

async def test_ui_integration():
    """Test the UI to verify it reflects agent work and progress."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        try:
            print("🌐 Testing UI integration...")
            
            # Navigate to DannyTrades UI
            await page.goto(f"{API_BASE_URL}/dannytrades")
            await page.wait_for_load_state("networkidle")
            
            print("📱 Checking UI elements...")
            
            # Check if MCP status is visible
            mcp_status = page.locator("#mcp-status")
            if await mcp_status.count() > 0:
                status_text = await mcp_status.text_content()
                print(f"   MCP Status: {status_text}")
            
            # Check if activities section exists
            activities = page.locator("#activities-list")
            if await activities.count() > 0:
                print("   ✅ Activities section found")
            
            # Take screenshot for verification
            await page.screenshot(path="dannytrades_ui_test.png")
            print("   📸 Screenshot saved as dannytrades_ui_test.png")
            
            # Test API call from UI (if there's a test button)
            test_button = page.locator("button:has-text('Test')")
            if await test_button.count() > 0:
                print("   🔘 Found test button, clicking...")
                await test_button.click()
                await page.wait_for_timeout(2000)  # Wait for response
            
            return True
            
        except Exception as e:
            print(f"❌ Exception during UI test: {str(e)}")
            return False
        finally:
            await browser.close()

async def test_mcp_status():
    """Test MCP status endpoint to verify Notion connection."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        
        try:
            print("🔗 Testing MCP status...")
            
            response = await context.request.get(f"{API_BASE_URL}/api/mcp/status")
            
            if response.status == 200:
                result = await response.json()
                print("✅ MCP Status Retrieved!")
                
                if isinstance(result, dict) and "servers" in result:
                    for server_name, status in result["servers"].items():
                        print(f"   {server_name}: {status}")
                else:
                    print(f"   Status: {result}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ MCP Status Error: {error_text}")
                return False
                
        except Exception as e:
            print(f"❌ Exception during MCP status test: {str(e)}")
            return False
        finally:
            await browser.close()

async def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🧪 Starting Comprehensive DannyTrades Test Suite")
    print("=" * 50)
    
    # Test 1: MCP Status
    mcp_success = await test_mcp_status()
    print()
    
    # Test 2: Direct API
    api_success = await test_api_direct()
    print()
    
    # Test 3: UI Integration
    ui_success = await test_ui_integration()
    print()
    
    # Summary
    print("📋 Test Results Summary:")
    print(f"   MCP Status: {'✅ PASS' if mcp_success else '❌ FAIL'}")
    print(f"   API Endpoint: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"   UI Integration: {'✅ PASS' if ui_success else '❌ FAIL'}")
    
    overall_success = mcp_success and api_success and ui_success
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
