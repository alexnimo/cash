#!/usr/bin/env python3
"""
Complete integration test for Z.AI + Direct Notion API.
Tests the full flow: Z.AI analysis → Direct Notion API save.
"""

import asyncio
import httpx
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_complete_integration():
    """Test complete Z.AI + Notion integration."""
    
    print("🚀 Starting complete Z.AI + Notion integration test...")
    
    # Test data
    test_data = {
        "symbol": "TSLA",
        "price": 250.75,
        "change": 3.2,
        "volume": 45000000
    }
    
    # Step 1: Z.AI Analysis
    print("\n📊 Step 1: Getting Z.AI analysis...")
    
    # Z.AI configuration
    api_key = "19f034ce9ef5405691533ef5756eff21.IUiinRQBx2rYJBoL"
    base_url = "https://api.z.ai/api/coding/paas/v4"
    model = "glm-4.5"
    
    # Create analysis prompt
    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst. Provide detailed stock analysis with investment recommendations."
        },
        {
            "role": "user", 
            "content": f"""Analyze {test_data['symbol']} stock:
- Current Price: ${test_data['price']}
- Daily Change: {test_data['change']}%
- Volume: {test_data['volume']:,} shares

Provide analysis including market sentiment, technical indicators, and investment recommendation."""
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "opencode/1.0"
    }
    
    zai_start = time.time()
    analysis_content = ""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0  # Increased timeout
            )
            
            zai_time = time.time() - zai_start
            
            if response.status_code == 200:
                result_data = response.json()
                analysis_content = result_data['choices'][0]['message']['content']
                
                print(f"✅ Z.AI analysis completed in {zai_time:.2f}s")
                print(f"📝 Analysis length: {len(analysis_content)} characters")
                print(f"💭 Preview: {analysis_content[:150]}...")
                
                if 'usage' in result_data:
                    usage = result_data['usage']
                    print(f"📊 Tokens: {usage.get('total_tokens', 'N/A')}")
            else:
                print(f"❌ Z.AI failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Z.AI exception: {e}")
        return False
    
    # Step 2: Save to Notion
    print("\n💾 Step 2: Saving to Notion database...")
    
    notion_token = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    if not notion_token:
        print("❌ No NOTION_API_KEY found in environment")
        return False
        
    if not database_id:
        print("❌ No NOTION_DATABASE_ID found in environment")
        return False
    
    # Prepare Notion page data
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    page_title = f"{test_data['symbol']} Analysis - {timestamp}"
    
    # Create page properties (using standard Notion title property)
    properties = {
        "title": {"title": [{"text": {"content": page_title}}]}
    }
    
    payload = {
        "parent": {"database_id": database_id},
        "properties": properties
    }
    
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    notion_start = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.notion.com/v1/pages",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            notion_time = time.time() - notion_start
            
            print(f"📡 Notion API response: {response.status_code}")
            
            if response.status_code == 200:
                result_data = response.json()
                page_id = result_data.get('id', 'Unknown')
                
                print(f"✅ Notion page created in {notion_time:.2f}s")
                print(f"📄 Page ID: {page_id[:8]}...")
                print(f"🔗 Title: {page_title}")
                
                return True
            else:
                error_text = response.text
                print(f"❌ Notion failed: {response.status_code}")
                print(f"🚨 Error: {error_text}")
                return False
                
    except Exception as e:
        print(f"❌ Notion exception: {e}")
        return False

async def main():
    """Main test function."""
    print("=" * 80)
    print("🧪 COMPLETE Z.AI + NOTION INTEGRATION TEST")
    print("=" * 80)
    
    start_time = time.time()
    success = await test_complete_integration()
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    if success:
        print(f"🎉 COMPLETE INTEGRATION SUCCESS! ({total_time:.2f}s total)")
        print("✅ Z.AI analysis → Direct Notion API save working perfectly!")
    else:
        print(f"💥 INTEGRATION FAILED! ({total_time:.2f}s total)")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
