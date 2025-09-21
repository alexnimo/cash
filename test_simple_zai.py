#!/usr/bin/env python3
"""
Simple Z.AI API test to verify connectivity and response without MCP integration.
"""

import asyncio
import httpx
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_zai_api():
    """Test Z.AI API directly without any MCP integration."""
    
    # Z.AI configuration
    api_key = "19f034ce9ef5405691533ef5756eff21.IUiinRQBx2rYJBoL"  # From config
    base_url = "https://api.z.ai/api/coding/paas/v4"
    model = "glm-4.5"
    
    # Simple test data
    test_data = {
        "symbol": "AAPL",
        "price": 150.25,
        "change": 2.5
    }
    
    # Create simple prompt
    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst. Provide brief analysis in 2-3 sentences."
        },
        {
            "role": "user", 
            "content": f"Analyze {test_data['symbol']} stock at ${test_data['price']} with {test_data['change']}% change."
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "opencode/1.0"
    }
    
    print(f"🚀 Testing Z.AI API...")
    print(f"📡 Endpoint: {base_url}/chat/completions")
    print(f"🤖 Model: {model}")
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=15.0  # Shorter timeout
            )
            
            processing_time = time.time() - start_time
            
            print(f"📡 Response Status: {response.status_code}")
            print(f"🕰️ Processing Time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                result_data = response.json()
                content = result_data['choices'][0]['message']['content']
                
                print(f"✅ SUCCESS!")
                print(f"📝 Response Length: {len(content)} characters")
                print(f"💭 Content Preview: {content[:150]}...")
                
                # Show usage if available
                if 'usage' in result_data:
                    usage = result_data['usage']
                    print(f"📊 Token Usage: {usage}")
                
                return True, content
            else:
                error_text = response.text
                print(f"❌ FAILED!")
                print(f"🚨 Error: {error_text}")
                return False, error_text
                
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ EXCEPTION after {processing_time:.2f}s!")
        print(f"🚨 Error: {str(e)}")
        return False, str(e)

async def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 SIMPLE Z.AI API TEST")
    print("=" * 60)
    
    success, result = await test_zai_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Z.AI API is working correctly!")
    else:
        print("💥 Z.AI API test failed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
