#!/usr/bin/env python3
"""
Debug configuration loading to understand why custom endpoint is None.
"""

import yaml
from app.core.config import load_config

def debug_config():
    """Debug configuration loading"""
    print("=== Configuration Debug ===")
    
    # Load raw config
    raw_config = load_config()
    print(f"Raw config keys: {list(raw_config.keys())}")
    
    # Check agents section first (this is where dannytrades is nested)
    agents = raw_config.get('agents', {})
    print(f"Agents keys: {list(agents.keys())}")
    
    # Check dannytrades section under agents
    dannytrades = agents.get('dannytrades', {})
    print(f"DannyTrades keys: {list(dannytrades.keys())}")
    
    # Check providers section
    providers = dannytrades.get('providers', {})
    print(f"Providers: {list(providers.keys())}")
    
    # Check OpenAI provider
    openai_provider = providers.get('openai', {})
    print(f"OpenAI provider config: {openai_provider}")
    print(f"OpenAI endpoint: {openai_provider.get('endpoint')}")
    
    # Load with DannyTradesConfig
    from app.agents.dannytrades.config import DannyTradesConfig
    config = DannyTradesConfig()
    print(f"Config _providers_config: {getattr(config, '_providers_config', 'NOT SET')}")
    print(f"Config _dannytrades_config: {getattr(config, '_dannytrades_config', 'NOT SET')}")
    
    # Test get_provider_config
    openai_config = config.get_provider_config("openai")
    print(f"get_provider_config result: {openai_config}")
    
    # Debug the get_provider_config method step by step
    print(f"Default provider: {config.default_provider}")
    print(f"Default model: {config.default_model}")
    provider_config = config._providers_config.get("openai", {})
    print(f"Direct provider config access: {provider_config}")

if __name__ == "__main__":
    debug_config()
