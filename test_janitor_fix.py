#!/usr/bin/env python3
"""
Test script to verify Janitor service cleanup preview fixes
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.janitor_service import janitor_service
from app.core.config import get_settings

async def test_janitor_preview():
    """Test the Janitor service cleanup preview functionality"""
    print("Testing Janitor Service Cleanup Preview...")
    
    # Get current status
    status = janitor_service.get_status()
    print(f"Current status: {status}")
    
    # Update runtime config to test with short retention and no recent file preservation
    print("\nUpdating runtime config...")
    config_update = {
        "retention_hours": 1,  # Very short retention for testing
        "preserve_recent_files": False,  # Don't preserve recent files
        "dry_run": True,  # Enable dry run for safety
        "log_deletions": True
    }
    
    updated_status = janitor_service.update_runtime_config(config_update)
    print(f"Updated status: {updated_status}")
    
    # Get effective config to verify runtime overrides are applied
    effective_config = janitor_service.get_effective_config()
    print(f"\nEffective config: {effective_config}")
    
    # Run cleanup preview
    print("\nRunning cleanup preview...")
    preview_result = await janitor_service.get_cleanup_preview()
    print(f"Preview result: {preview_result}")
    
    # Check if files were found
    total_files = preview_result.get('files_deleted', 0)
    if total_files > 0:
        print(f"✅ SUCCESS: Preview found {total_files} files to delete")
        for path_info in preview_result.get('paths_processed', []):
            print(f"  - {path_info['path']}: {path_info['files_deleted']} files")
    else:
        print("❌ ISSUE: Preview still shows 0 files")
        
        # Debug: Check if paths exist and have files
        settings = get_settings()
        base_path = Path(settings.storage.base_path)
        print(f"\nDebugging paths from base: {base_path}")
        
        for cleanup_path in effective_config['cleanup_paths']:
            full_path = base_path / cleanup_path
            print(f"  Checking path: {full_path}")
            if full_path.exists():
                files = list(full_path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                print(f"    Exists: Yes, Files: {file_count}")
                if file_count > 0:
                    # Show first few files for debugging
                    for i, f in enumerate([f for f in files if f.is_file()][:3]):
                        print(f"      - {f} (modified: {f.stat().st_mtime})")
            else:
                print(f"    Exists: No")

if __name__ == "__main__":
    asyncio.run(test_janitor_preview())
