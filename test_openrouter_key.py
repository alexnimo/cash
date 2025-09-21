#!/usr/bin/env python3
"""
Quick test script to verify OpenRouter API key validity
"""
import requests
import json

# Read the API key from config
with open('.cash_config/dannytrades_config.json', 'r') as f:
    config = json.load(f)

api_key = config['providers']['openai']['api_key']
endpoint = config['providers']['openai']['endpoint']

print(f"Testing OpenRouter API key: {api_key[:8]}...{api_key[-4:]}")
print(f"Endpoint: {endpoint}")

# Test with a simple request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "nvidia/nemotron-nano-9b-v2:free",
    "messages": [
        {"role": "user", "content": "Hello, this is a test"}
    ],
    "max_tokens": 10
}

try:
    response = requests.post(f"{endpoint}/chat/completions", 
                           headers=headers, 
                           json=data, 
                           timeout=10)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 401:
        print("\n❌ API Key is INVALID or EXPIRED")
        print("Please generate a new API key from: https://openrouter.ai/keys")
    elif response.status_code == 200:
        print("\n✅ API Key is VALID")
    else:
        print(f"\n⚠️  Unexpected response: {response.status_code}")
        
except Exception as e:
    print(f"Error testing API key: {e}")
