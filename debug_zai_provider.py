#!/usr/bin/env python3
"""
Debug script to check Z.AI provider status and routing.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.dannytrades.simple_agent import simple_agent

def debug_zai_provider():
    """Debug Z.AI provider configuration and routing."""
    print("🔍 Z.AI Provider Debug")
    print("=" * 40)
    
    # Check current provider
    print(f"Current provider: '{simple_agent.current_provider}'")
    print(f"Current model: {getattr(simple_agent.current_model, 'id', 'unknown')}")
    print(f"Model type: {type(simple_agent.current_model)}")
    
    # Check Z.AI routing condition
    is_zai = simple_agent.current_provider == "zai"
    has_model = simple_agent.current_model is not None
    
    print(f"\nZ.AI Routing Check:")
    print(f"  Provider == 'zai': {is_zai}")
    print(f"  Has model: {has_model}")
    print(f"  Would use Z.AI direct: {is_zai and has_model}")
    
    # Check config
    print(f"\nConfig default provider: '{simple_agent.config.default_provider}'")
    
    # Check Z.AI config specifically
    try:
        zai_config = simple_agent.config.get_provider_config('zai')
        print(f"\nZ.AI Config:")
        print(f"  Enabled: {zai_config.get('enabled', False)}")
        print(f"  Model: {zai_config.get('model_id', 'N/A')}")
        print(f"  Has API key: {'api_key' in zai_config and bool(zai_config.get('api_key'))}")
    except Exception as e:
        print(f"\n❌ Error getting Z.AI config: {e}")

if __name__ == "__main__":
    debug_zai_provider()
