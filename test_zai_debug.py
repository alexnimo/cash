#!/usr/bin/env python3
"""
Debug Z.AI API to understand why it returns empty content in production.
"""

import asyncio
import httpx
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def debug_zai_api():
    """Debug Z.AI API with different configurations."""
    
    # Z.AI configuration
    api_key = "19f034ce9ef5405691533ef5756eff21.IUiinRQBx2rYJBoL"
    base_url = "https://api.z.ai/api/coding/paas/v4"
    model = "glm-4.5"
    
    test_cases = [
        {
            "name": "Simple Test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            "max_tokens": 50
        },
        {
            "name": "Stock Analysis Test",
            "messages": [
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": "Analyze AAPL stock at $150.25 with 2.5% change."}
            ],
            "max_tokens": 200
        },
        {
            "name": "Production-like Test",
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst specializing in stock market analysis and investment recommendations. Your role is to provide comprehensive, data-driven insights on stocks, market trends, and investment opportunities."},
                {"role": "user", "content": "Analyze LMND stock: Price $15.50, Change 3.2%. Provide brief investment analysis."}
            ],
            "max_tokens": 500
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        payload = {
            "model": model,
            "messages": test_case["messages"],
            "stream": False,
            "max_tokens": test_case["max_tokens"],
            "temperature": 0.3
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "opencode/1.0",
            "X-Client-Type": "coding-agent",
            "X-Agent-Name": "opencode",
            "X-Agent-Version": "1.0"
        }
        
        print(f"📝 Messages: {len(test_case['messages'])}")
        print(f"🎯 Max tokens: {test_case['max_tokens']}")
        print(f"🌡️ Temperature: 0.3")
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0
                )
                
                processing_time = time.time() - start_time
                
                print(f"📡 Status: {response.status_code}")
                print(f"🕰️ Time: {processing_time:.2f}s")
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Debug the response structure
                    print(f"📦 Response keys: {list(result_data.keys())}")
                    
                    if 'choices' in result_data and len(result_data['choices']) > 0:
                        choice = result_data['choices'][0]
                        print(f"🎯 Choice keys: {list(choice.keys())}")
                        
                        if 'message' in choice:
                            message = choice['message']
                            print(f"💬 Message keys: {list(message.keys())}")
                            
                            content = message.get('content', '')
                            print(f"📝 Content length: {len(content)} chars")
                            
                            if content:
                                print(f"✅ Content preview: {content[:100]}...")
                            else:
                                print(f"❌ EMPTY CONTENT!")
                                print(f"🔍 Full message: {json.dumps(message, indent=2)}")
                        else:
                            print(f"❌ No 'message' in choice!")
                            print(f"🔍 Full choice: {json.dumps(choice, indent=2)}")
                    else:
                        print(f"❌ No choices in response!")
                        print(f"🔍 Full response: {json.dumps(result_data, indent=2)}")
                    
                    # Show usage if available
                    if 'usage' in result_data:
                        usage = result_data['usage']
                        print(f"📊 Usage: {usage}")
                else:
                    print(f"❌ Error: {response.status_code}")
                    print(f"🚨 Response: {response.text}")
                    
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Exception: {e}")
            print(f"🕰️ Failed after: {processing_time:.2f}s")

async def main():
    """Main debug function."""
    print("🔍 Z.AI API DEBUG SESSION")
    print("=" * 60)
    
    await debug_zai_api()
    
    print("\n" + "=" * 60)
    print("🔍 Debug session complete")

if __name__ == "__main__":
    asyncio.run(main())
