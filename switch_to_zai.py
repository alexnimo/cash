#!/usr/bin/env python3
"""
Script to switch the DannyTrades agent to Z.AI provider.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.dannytrades.simple_agent import simple_agent

async def switch_to_zai():
    """Switch the agent to Z.AI provider."""
    print("🔄 Switching to Z.AI provider...")
    
    # Get Z.AI API key from environment
    zai_api_key = os.getenv("ZAI_API_KEY")
    if not zai_api_key:
        print("❌ ZAI_API_KEY environment variable not set")
        print("Please set your Z.AI API key: export ZAI_API_KEY='your_key_here'")
        return False
    
    try:
        # Switch to Z.AI provider
        success = await simple_agent.switch_provider(
            provider_name="zai",
            api_key=zai_api_key,
            model="glm-4.5",
            endpoint="https://api.z.ai/api/coding/paas/v4",
            user_agent="opencode/1.0"
        )
        
        if success:
            print("✅ Successfully switched to Z.AI provider")
            print(f"   Current provider: {simple_agent.current_provider}")
            print(f"   Current model: {getattr(simple_agent.current_model, 'id', 'unknown')}")
            
            # Test the Z.AI direct processing path
            test_data = {
                "symbol": "AAPL",
                "price": 150.25,
                "change": 2.5,
                "volume": 45000000
            }
            
            print(f"\n📊 Testing Z.AI processing...")
            result = await simple_agent.process_stock_data(test_data)
            
            if result.success:
                print(f"✅ Z.AI processing successful!")
                print(f"   Processing time: {result.processing_time:.2f}s")
                print(f"   Analysis preview: {(result.reasoning or '')[:100]}...")
            else:
                print(f"❌ Z.AI processing failed: {result.error_message}")
            
            return success
        else:
            print("❌ Failed to switch to Z.AI provider")
            return False
            
    except Exception as e:
        print(f"💥 Error switching to Z.AI: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(switch_to_zai())
    sys.exit(0 if success else 1)
