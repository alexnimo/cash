#!/usr/bin/env python3
"""
Debug script to check why Z.AI direct processing isn't being used in production.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.dannytrades.simple_agent import simple_agent

async def debug_production_zai():
    """Debug Z.AI routing in production environment."""
    print("🔍 Production Z.AI Debug")
    print("=" * 40)
    
    # Check current state
    print(f"Current provider: '{simple_agent.current_provider}'")
    print(f"Provider type: {type(simple_agent.current_provider)}")
    print(f"Provider repr: {repr(simple_agent.current_provider)}")
    print(f"Provider bytes: {simple_agent.current_provider.encode('utf-8') if simple_agent.current_provider else None}")
    print(f"Current model: {getattr(simple_agent.current_model, 'id', 'unknown')}")
    print(f"Model type: {type(simple_agent.current_model)}")
    
    # Check the exact condition with detailed comparison
    is_zai = simple_agent.current_provider == "zai"
    has_model = simple_agent.current_model is not None
    
    print(f"\nDetailed Condition Check:")
    print(f"  simple_agent.current_provider: '{simple_agent.current_provider}'")
    print(f"  'zai': 'zai'")
    print(f"  Length comparison: {len(simple_agent.current_provider)} vs {len('zai')}")
    print(f"  Equality check: {simple_agent.current_provider == 'zai'}")
    print(f"  Has model: {has_model}")
    print(f"  Combined condition: {is_zai and has_model}")
    
    # Check if there are any hidden characters
    if simple_agent.current_provider:
        for i, char in enumerate(simple_agent.current_provider):
            print(f"    Char {i}: '{char}' (ord: {ord(char)})")
    
    # Also check config default provider
    print(f"\nConfig default provider: '{simple_agent.config.default_provider}'")
    
    # Test with actual stock data like production
    test_data = {
        "symbol": "LMND",
        "price": 100.0,
        "change": 2.5,
        "volume": 1000000
    }
    
    print(f"\n📊 Testing with production-like data:")
    print(f"  Symbol: {test_data['symbol']}")
    
    # Simulate the exact path the production code takes
    try:
        result = await simple_agent.process_stock_data(test_data)
        print(f"\nResult:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Processing time: {result.processing_time:.3f}s")
            print(f"  Analysis length: {len(result.reasoning or '')} chars")
        else:
            print(f"  Error: {result.error_message}")
    except Exception as e:
        print(f"\n💥 Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_production_zai())
