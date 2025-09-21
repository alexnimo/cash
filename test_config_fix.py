#!/usr/bin/env python3
"""Test script to verify DannyTradesConfig attribute fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test that DannyTradesConfig loads without attribute errors"""
    try:
        from app.agents.dannytrades.config import DannyTradesConfig
        
        print("Loading DannyTradesConfig...")
        config = DannyTradesConfig()
        
        print(f"✅ Config loaded successfully")
        print(f"✅ Default provider: {config.default_provider}")
        print(f"✅ Default model: {config.default_model}")
        print(f"✅ Has _dannytrades_config attribute: {hasattr(config, '_dannytrades_config')}")
        print(f"✅ Has _providers_config attribute: {hasattr(config, '_providers_config')}")
        
        # Test MCP server retrieval
        mcp_servers = config.get_mcp_servers()
        print(f"✅ MCP servers loaded: {len(mcp_servers)} servers")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_initialization():
    """Test that DannyTradesAgnoAgent can be created without errors"""
    try:
        from dannytrades_agno import DannyTradesAgnoAgent
        
        print("\nCreating DannyTradesAgnoAgent...")
        agent = DannyTradesAgnoAgent()
        
        print(f"✅ Agent created successfully")
        print(f"✅ Current provider: {agent.current_provider}")
        print(f"✅ Config loaded: {agent.config is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing DannyTrades configuration fix...\n")
    
    config_ok = test_config_loading()
    agent_ok = test_agent_initialization()
    
    if config_ok and agent_ok:
        print("\n🎉 All tests passed! The attribute error has been fixed.")
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
