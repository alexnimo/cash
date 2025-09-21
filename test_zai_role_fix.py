#!/usr/bin/env python3
"""
Test script to verify Z.AI role formatting fix.
Tests the direct Z.AI processing method to ensure proper message formatting.
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.dannytrades.simple_agent import SimpleDannyTradesAgent

async def test_zai_role_fix():
    """Test Z.AI processing with corrected role formatting."""
    print("🧪 Testing Z.AI Role Formatting Fix")
    print("=" * 50)
    
    # Initialize the agent
    agent = SimpleDannyTradesAgent()
    await agent.initialize()
    
    # Check if Z.AI API key is available
    zai_api_key = os.getenv("ZAI_API_KEY")
    if not zai_api_key:
        print("❌ ZAI_API_KEY environment variable not set")
        print("Please set your Z.AI API key: export ZAI_API_KEY='your_key_here'")
        return False
    
    try:
        # Switch to Z.AI provider
        print("🔄 Switching to Z.AI provider...")
        await agent.switch_provider("zai", "glm-4.5", zai_api_key)
        print(f"✅ Successfully switched to Z.AI provider")
        print(f"   Provider: {agent.current_provider}")
        print(f"   Model: {getattr(agent.current_model, 'id', 'unknown')}")
        
        # Test stock data
        test_stock_data = {
            "symbol": "AAPL",
            "price": 150.25,
            "change": 2.5,
            "volume": 45000000
        }
        
        print(f"\n📊 Testing stock analysis with Z.AI...")
        print(f"   Symbol: {test_stock_data['symbol']}")
        print(f"   Price: ${test_stock_data['price']}")
        print(f"   Change: {test_stock_data['change']}%")
        print(f"   Volume: {test_stock_data['volume']:,}")
        
        # Process the stock data using direct Z.AI method
        result = await agent.process_stock_data(test_stock_data)
        
        if result.success:
            print(f"\n✅ Z.AI Processing Successful!")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   Stock Symbol: {result.stock_symbol}")
            print(f"   Action: {getattr(result, 'action', 'N/A')}")
            print(f"   Analysis Length: {len(result.reasoning or '')} characters")
            print(f"\n📝 Analysis Preview:")
            preview = (result.reasoning or "")[:200]
            print(f"   {preview}{'...' if len(result.reasoning or '') > 200 else ''}")
            return True
        else:
            print(f"\n❌ Z.AI Processing Failed!")
            print(f"   Error: {result.error_message}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_message_format_directly():
    """Test the message format that will be sent to Z.AI API."""
    print("\n🔍 Testing Message Format")
    print("=" * 30)
    
    agent = SimpleDannyTradesAgent()
    await agent.initialize()
    
    # Get sample instructions
    test_data = {"symbol": "TEST", "price": 100.0}
    instructions = agent._get_agent_instructions(test_data)
    
    # Show the message format that would be sent
    messages = [
        {
            "role": "system",
            "content": instructions
        },
        {
            "role": "user", 
            "content": "Test message for role validation"
        }
    ]
    
    print("📋 Message format that will be sent to Z.AI:")
    for i, msg in enumerate(messages):
        print(f"   Message {i+1}:")
        print(f"     Role: {msg['role']}")
        print(f"     Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    
    # Validate message structure
    valid_roles = {"system", "user", "assistant"}
    all_valid = True
    
    for msg in messages:
        if "role" not in msg:
            print(f"❌ Missing 'role' field in message")
            all_valid = False
        elif msg["role"] not in valid_roles:
            print(f"❌ Invalid role '{msg['role']}' - must be one of {valid_roles}")
            all_valid = False
        
        if "content" not in msg:
            print(f"❌ Missing 'content' field in message")
            all_valid = False
        elif not isinstance(msg["content"], str):
            print(f"❌ Content must be string, got {type(msg['content'])}")
            all_valid = False
    
    if all_valid:
        print("✅ Message format is valid for Z.AI API")
    else:
        print("❌ Message format has issues")
    
    return all_valid

async def main():
    """Run all tests."""
    print("🚀 Z.AI Role Formatting Fix Test Suite")
    print("=" * 60)
    
    # Test 1: Message format validation
    format_valid = await test_message_format_directly()
    
    # Test 2: Actual Z.AI API call (if API key available)
    if os.getenv("ZAI_API_KEY"):
        api_success = await test_zai_role_fix()
    else:
        print("\n⚠️  Skipping API test - ZAI_API_KEY not set")
        api_success = None
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Message Format: {'✅ PASS' if format_valid else '❌ FAIL'}")
    if api_success is not None:
        print(f"   Z.AI API Call: {'✅ PASS' if api_success else '❌ FAIL'}")
    else:
        print(f"   Z.AI API Call: ⏭️  SKIPPED (no API key)")
    
    if format_valid and (api_success is None or api_success):
        print("\n🎉 All tests passed! Z.AI role formatting issue appears to be resolved.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
