#!/usr/bin/env python3
"""
Test script for Janitor service API endpoints using MCP tools
"""
import asyncio
import sys
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_janitor_api():
    """Test all Janitor service API endpoints"""
    try:
        # Import required modules
        from app.api.routes import router
        from app.services.janitor_service import janitor_service
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        # Create test app
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        print("🧪 Testing Janitor Service API Endpoints")
        print("=" * 50)
        
        # Test 1: Get Status
        print("\n1. Testing GET /api/janitor/status")
        response = client.get("/api/janitor/status")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status retrieved successfully")
            print(f"   - Enabled: {data['data']['enabled']}")
            print(f"   - Running: {data['data']['running']}")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test 2: Update Config
        print("\n2. Testing POST /api/janitor/config")
        test_config = {
            "retention_hours": 48,
            "dry_run": True,
            "log_deletions": True
        }
        response = client.post("/api/janitor/config", json=test_config)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Config updated successfully")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test 3: Get Preview
        print("\n3. Testing GET /api/janitor/cleanup/preview")
        response = client.get("/api/janitor/cleanup/preview")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Preview generated successfully")
            print(f"   - Would delete: {data['data']['files_deleted']} files")
            print(f"   - Would free: {data['data']['bytes_freed']} bytes")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test 4: Manual Cleanup (dry run should be enabled)
        print("\n4. Testing POST /api/janitor/cleanup/manual")
        response = client.post("/api/janitor/cleanup/manual")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Manual cleanup completed")
            print(f"   - Files processed: {data['data']['files_deleted']}")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test 5: Start Service
        print("\n5. Testing POST /api/janitor/start")
        response = client.post("/api/janitor/start")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Service started successfully")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test 6: Stop Service
        print("\n6. Testing POST /api/janitor/stop")
        response = client.post("/api/janitor/stop")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Service stopped successfully")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        print("\n" + "=" * 50)
        print("🎉 API endpoint testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_janitor_api())
    sys.exit(0 if success else 1)
