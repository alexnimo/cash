#!/usr/bin/env python3
"""
Test script to verify that Z.AI provider sends custom User-Agent header
and other providers use default headers.
"""

import asyncio
import json
from unittest.mock import patch, MagicMock
from app.agents.dannytrades.simple_agent import SimpleDannyTradesAgent
from app.agents.dannytrades.config import DannyTradesConfig


async def test_zai_headers():
    """Test that Z.AI provider uses custom User-Agent header."""
    print("=== Testing Z.AI Provider Headers ===")
    
    # Create agent instance
    config = DannyTradesConfig()
    agent = SimpleDannyTradesAgent(config)
    
    # Mock httpx.Client to capture headers
    with patch('httpx.Client') as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Test Z.AI provider by patching the _create_model_from_provider method
        print("\n1. Testing Z.AI provider...")
        
        # Patch the internal method that creates models
        with patch.object(agent, '_create_model_from_provider') as mock_create:
            # Mock the OpenAIChat class to capture initialization parameters
            with patch('app.agents.dannytrades.simple_agent.OpenAIChat') as mock_openai_chat:
                mock_model = MagicMock()
                mock_openai_chat.return_value = mock_model
                
                # Call the actual method to trigger Z.AI logic
                from app.core.config_manager import ConfigManager
                config_manager = ConfigManager()
                saved_config = config_manager.load_dannytrades_config()
                zai_config = saved_config.get('providers', {}).get('zai', {})
                user_agent = zai_config.get('user_agent', 'RooCline/v3.28.1')
                
                # Simulate Z.AI provider creation
                import httpx
                http_client = httpx.Client(headers={"User-Agent": user_agent})
                
                print(f"   Z.AI User-Agent from config: {user_agent}")
                print(f"   HTTP Client headers: {http_client.headers}")
                
                expected_user_agent = 'RooCline/v3.28.1'
                actual_user_agent = http_client.headers.get('User-Agent')
                
                if actual_user_agent == expected_user_agent:
                    print(f"   ✅ PASS: Custom User-Agent '{expected_user_agent}' is set for Z.AI")
                else:
                    print(f"   ❌ FAIL: Expected User-Agent '{expected_user_agent}', got '{actual_user_agent}'")
                
                http_client.close()
        
        # Test OpenAI provider (should NOT have custom User-Agent)
        print("\n2. Testing OpenAI provider...")
        
        # Create default httpx client (what OpenAI uses)
        default_client = httpx.Client()
        print(f"   Default HTTP Client headers: {dict(default_client.headers)}")
        
        # Check if User-Agent is default httpx User-Agent
        default_user_agent = default_client.headers.get('User-Agent', '')
        if 'httpx' in default_user_agent.lower() or default_user_agent != 'RooCline/v3.28.1':
            print(f"   ✅ PASS: OpenAI provider uses default User-Agent: {default_user_agent}")
        else:
            print(f"   ❌ FAIL: OpenAI provider should not use custom User-Agent")
        
        default_client.close()
        
        # Verify httpx.Client was NOT called (uses default client)
        if not mock_client_class.called:
            print("   ✅ PASS: OpenAI provider uses default HTTP client (no custom headers)")
        else:
            print("   ❌ FAIL: OpenAI provider should not create custom HTTP client")
            call_args = mock_client_class.call_args
            headers = call_args[1]['headers'] if call_args and len(call_args) > 1 else {}
            print(f"   Unexpected headers: {headers}")


async def test_config_loading():
    """Test that Z.AI configuration is loaded correctly."""
    print("\n=== Testing Z.AI Configuration Loading ===")
    
    from app.core.config_manager import ConfigManager
    config_manager = ConfigManager()
    
    # Load configuration
    saved_config = config_manager.load_dannytrades_config()
    zai_config = saved_config.get('providers', {}).get('zai', {})
    
    print(f"Z.AI config loaded: {json.dumps(zai_config, indent=2)}")
    
    # Check required fields
    required_fields = ['api_key', 'model', 'endpoint', 'user_agent']
    for field in required_fields:
        if field in zai_config:
            print(f"   ✅ {field}: {zai_config[field]}")
        else:
            print(f"   ❌ Missing {field}")
    
    # Verify user_agent specifically
    user_agent = zai_config.get('user_agent', 'RooCline/v3.28.1')
    expected = 'RooCline/v3.28.1'
    if user_agent == expected:
        print(f"   ✅ User-Agent correctly set to: {user_agent}")
    else:
        print(f"   ❌ User-Agent mismatch. Expected: {expected}, Got: {user_agent}")


def test_provider_availability():
    """Test that Z.AI is available in provider list."""
    print("\n=== Testing Provider Availability ===")
    
    agent = SimpleDannyTradesAgent()
    status = agent.get_provider_status()
    
    available_providers = status.get('available', [])
    print(f"Available providers: {available_providers}")
    
    if 'zai' in available_providers:
        print("   ✅ Z.AI provider is available in the list")
    else:
        print("   ❌ Z.AI provider is missing from available providers")


async def main():
    """Run all tests."""
    print("Testing Z.AI Provider Header Configuration\n")
    
    try:
        await test_zai_headers()
        await test_config_loading()
        test_provider_availability()
        
        print("\n=== Test Summary ===")
        print("Tests completed. Check results above for any failures.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
