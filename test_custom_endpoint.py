#!/usr/bin/env python3
"""
Test script to verify custom OpenAI endpoint configuration works.
"""

import os
import sys
import asyncio
from app.agents.dannytrades.config import DannyTradesConfig
from app.agents.dannytrades._setup_llm_model import setup_llm_model

async def test_custom_endpoint():
    """Test custom OpenAI endpoint configuration"""
    try:
        print("Testing custom OpenAI endpoint configuration...")
        
        # Load config
        config = DannyTradesConfig()
        print(f"✓ Config loaded successfully")
        print(f"  Provider: {config.default_provider}")
        print(f"  Model: {config.default_model}")
        
        # Get OpenAI provider config
        openai_config = config.get_provider_config("openai")
        print(f"  OpenAI Config: {openai_config}")
        print(f"  Custom endpoint: {openai_config.get('endpoint')}")
        print(f"  API key present: {bool(config.openai_api_key)}")
        print(f"  Providers config: {getattr(config, '_providers_config', {}).get('openai', {})}")
        print(f"  Raw providers config: {getattr(config, '_providers_config', {})}")
        
        # Check if we can access the config directly
        from app.core.config import load_config
        raw_config = load_config()
        dannytrades_config = raw_config.get('dannytrades', {})
        providers_config = dannytrades_config.get('providers', {})
        print(f"  Raw YAML providers: {providers_config.get('openai', {})}")
        
        # Test validation
        is_valid, error_msg = config.validate_llm_config()
        print(f"  Validation: {'✓ Valid' if is_valid else f'✗ Invalid: {error_msg}'}")
        
        # If using OpenAI provider, test model setup
        if config.default_provider == "openai":
            print("\nTesting OpenAI model setup...")
            try:
                model, provider = await setup_llm_model(config)
                print(f"✓ Model setup successful: {provider} - {model.id}")
                print(f"  Model type: {type(model).__name__}")
                if hasattr(model, 'base_url'):
                    print(f"  Base URL: {model.base_url}")
                if hasattr(model, 'api_key'):
                    print(f"  API Key: {'Set' if model.api_key else 'Not set'}")
            except Exception as e:
                print(f"✗ Model setup failed: {e}")
        else:
            print(f"\nSkipping OpenAI model test (current provider: {config.default_provider})")
        
        print("\n✓ Test completed successfully")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_custom_endpoint())
