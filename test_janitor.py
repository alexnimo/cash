#!/usr/bin/env python3
"""
Test script for the improved Janitor service
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_janitor_service():
    """Test the janitor service functionality"""
    try:
        from app.services.janitor_service import janitor_service
        from app.core.config import get_settings
        
        print("✅ Successfully imported janitor service")
        
        # Test configuration loading
        settings = get_settings()
        janitor_config = settings.janitor
        
        print(f"✅ Janitor config loaded:")
        print(f"   - Enabled: {janitor_config.enabled}")
        print(f"   - Schedule: {janitor_config.schedule}")
        print(f"   - Retention hours: {janitor_config.retention_hours}")
        print(f"   - Dry run: {janitor_config.dry_run}")
        print(f"   - Cleanup paths: {janitor_config.cleanup_paths}")
        print(f"   - File patterns: {janitor_config.file_patterns}")
        print(f"   - Max file size MB: {janitor_config.max_file_size_mb}")
        print(f"   - Exclude patterns: {janitor_config.exclude_patterns}")
        print(f"   - Log deletions: {janitor_config.log_deletions}")
        print(f"   - Preserve recent files: {janitor_config.preserve_recent_files}")
        
        # Test status retrieval
        status = janitor_service.get_status()
        print(f"✅ Service status retrieved:")
        print(f"   - Running: {status['running']}")
        print(f"   - Enabled: {status['enabled']}")
        
        # Test runtime config update
        test_config = {
            "dry_run": True,
            "retention_hours": 24
        }
        updated_status = janitor_service.update_runtime_config(test_config)
        print(f"✅ Runtime config updated successfully")
        
        # Test cleanup preview
        preview = await janitor_service.get_cleanup_preview()
        print(f"✅ Cleanup preview generated:")
        print(f"   - Would delete: {preview['files_deleted']} files")
        print(f"   - Would free: {preview['bytes_freed']} bytes")
        print(f"   - Errors: {len(preview.get('errors', []))}")
        
        print("\n🎉 All janitor service tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_janitor_service())
    sys.exit(0 if success else 1)
