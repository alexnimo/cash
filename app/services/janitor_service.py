"""
File Janitor Service - Automated cleanup of generated files
"""
import os
import glob
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from croniter import croniter
import time
import fnmatch

from app.core.config import get_settings

logger = logging.getLogger(__name__)

class JanitorService:
    """
    Automated file cleanup service that runs on a cron schedule
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.janitor_config = self.settings.janitor
        self.base_path = Path(self.settings.storage.base_path)
        self.is_running = False
        self.last_cleanup = None
        self._runtime_config = {}  # For UI overrides
        
    async def start(self):
        """Start the janitor service"""
        if not self.janitor_config.enabled:
            logger.info("Janitor service is disabled in configuration")
            return
            
        logger.info(f"Starting janitor service with schedule: {self.janitor_config.schedule}")
        self.is_running = True
        
        # Start the cleanup scheduler
        asyncio.create_task(self._scheduler())
        
    async def stop(self):
        """Stop the janitor service"""
        logger.info("Stopping janitor service")
        self.is_running = False
        
    async def _scheduler(self):
        """Main scheduler loop"""
        cron = croniter(self.janitor_config.schedule)
        
        while self.is_running:
            try:
                # Calculate next run time
                next_run = cron.get_next(datetime)
                current_time = datetime.now()
                
                # Wait until next scheduled time
                wait_seconds = (next_run - current_time).total_seconds()
                
                if wait_seconds > 0:
                    logger.info(f"Next cleanup scheduled for: {next_run}")
                    await asyncio.sleep(wait_seconds)
                
                if self.is_running:
                    await self.cleanup_files()
                    
            except Exception as e:
                logger.error(f"Error in janitor scheduler: {str(e)}", exc_info=True)
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)
                
    async def cleanup_files(self) -> Dict[str, Any]:
        """
        Perform file cleanup based on configuration
        Returns cleanup statistics
        """
        logger.info("Starting scheduled file cleanup")
        
        cleanup_stats = {
            "start_time": datetime.now().isoformat(),
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": [],
            "paths_processed": []
        }
        
        # Get effective configuration with runtime overrides
        effective_config = self.get_effective_config()
        cutoff_time = datetime.now() - timedelta(hours=effective_config["retention_hours"])
        
        try:
            # Process each cleanup path
            for cleanup_path in effective_config["cleanup_paths"]:
                full_path = self.base_path / cleanup_path
                
                if not full_path.exists():
                    logger.debug(f"Cleanup path does not exist: {full_path}")
                    continue
                    
                path_stats = await self._cleanup_path(full_path, cutoff_time, effective_config)
                cleanup_stats["files_deleted"] += path_stats["files_deleted"]
                cleanup_stats["bytes_freed"] += path_stats["bytes_freed"]
                cleanup_stats["errors"].extend(path_stats["errors"])
                cleanup_stats["paths_processed"].append({
                    "path": str(cleanup_path),
                    "files_deleted": path_stats["files_deleted"],
                    "bytes_freed": path_stats["bytes_freed"]
                })
                
        except Exception as e:
            error_msg = f"Error during file cleanup: {str(e)}"
            logger.error(error_msg, exc_info=True)
            cleanup_stats["errors"].append(error_msg)
            
        finally:
            cleanup_stats["end_time"] = datetime.now().isoformat()
            cleanup_stats["duration_seconds"] = (
                datetime.fromisoformat(cleanup_stats["end_time"]) - 
                datetime.fromisoformat(cleanup_stats["start_time"])
            ).total_seconds()
            
            self.last_cleanup = cleanup_stats
            
            # Log summary
            logger.info(
                f"Cleanup completed: {cleanup_stats['files_deleted']} files deleted, "
                f"{cleanup_stats['bytes_freed'] / (1024*1024):.2f} MB freed, "
                f"{len(cleanup_stats['errors'])} errors"
            )
            
        return cleanup_stats
        
    async def _cleanup_path(self, path: Path, cutoff_time: datetime, effective_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recursively delete files older than cutoff_time in the given path"""
        stats = {"files_deleted": 0, "bytes_freed": 0, "errors": []}
        
        logger.info(f"Processing cleanup path: {path}")
        logger.info(f"Cutoff time: {cutoff_time}")
        logger.info(f"Effective config: {effective_config}")
        
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return stats
            
        try:
            files_found = 0
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    files_found += 1
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        file_size = file_path.stat().st_size
                        
                        logger.debug(f"Found file: {file_path} (modified: {file_time}, size: {file_size})")
                        
                        # Check exclusion patterns
                        if self._should_exclude_file(file_path):
                            logger.debug(f"Excluding file due to patterns: {file_path}")
                            continue
                            
                        # Check if file should be preserved due to being recent
                        if self._should_preserve_recent(file_time, effective_config):
                            logger.debug(f"Preserving recent file: {file_path} (modified: {file_time})")
                            continue
                        
                        # Check if file is older than retention period
                        if file_time < cutoff_time:
                            if effective_config and effective_config.get("dry_run", False):
                                logger.info(f"[DRY RUN] Would delete: {file_path} ({file_size} bytes, modified: {file_time})")
                            else:
                                file_path.unlink()
                                if effective_config and effective_config.get("log_deletions", True):
                                    logger.info(f"Deleted file: {file_path} ({file_size} bytes)")
                                else:
                                    logger.debug(f"Deleted file: {file_path} ({file_size} bytes)")
                            
                            stats["files_deleted"] += 1
                            stats["bytes_freed"] += file_size
                        else:
                            logger.debug(f"File too recent, not deleting: {file_path} (modified: {file_time}, cutoff: {cutoff_time})")
                            
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        stats["errors"].append(str(e))
            
            logger.info(f"Total files found in {path}: {files_found}")
                        
        except Exception as e:
            logger.error(f"Error processing path {path}: {str(e)}")
            stats["errors"].append(str(e))
            
        return stats
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from cleanup"""
        for pattern in self.janitor_config.exclude_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
        return False
    
    def _exceeds_size_limit(self, file_size: int) -> bool:
        """Check if file exceeds size limit"""
        if self.janitor_config.max_file_size_mb is None:
            return False
        max_size_bytes = self.janitor_config.max_file_size_mb * 1024 * 1024
        return file_size > max_size_bytes
    
    def _should_preserve_recent(self, file_time: datetime, effective_config: Dict[str, Any] = None) -> bool:
        """Check if file should be preserved due to being recent"""
        preserve_recent = effective_config.get("preserve_recent_files", self.janitor_config.preserve_recent_files) if effective_config else self.janitor_config.preserve_recent_files
        if not preserve_recent:
            return False
        recent_cutoff = datetime.now() - timedelta(hours=24)
        return file_time > recent_cutoff
        
    def get_status(self) -> Dict[str, Any]:
        """Get current janitor service status"""
        return {
            "enabled": self.janitor_config.enabled,
            "running": self.is_running,
            "schedule": self.janitor_config.schedule,
            "last_cleanup": self.last_cleanup,
            "dry_run": self.janitor_config.dry_run,
            "config": {
                "cleanup_paths": self.janitor_config.cleanup_paths,
                "file_patterns": self.janitor_config.file_patterns,
                "retention_hours": self.janitor_config.retention_hours,
                "exclude_patterns": self.janitor_config.exclude_patterns,
                "log_deletions": self.janitor_config.log_deletions,
                "preserve_recent_files": self.janitor_config.preserve_recent_files
            },
            "runtime_config": self._runtime_config
        }
        
    async def manual_cleanup(self) -> Dict[str, Any]:
        """
        Trigger manual cleanup (useful for testing/admin)
        """
        logger.info("Manual cleanup triggered")
        return await self.cleanup_files()
    
    def update_runtime_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update runtime configuration (for UI control)
        """
        self._runtime_config.update(config_updates)
        logger.info(f"Runtime config updated: {config_updates}")
        return self.get_status()
    
    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get effective configuration (base config + runtime overrides)
        """
        config = {
            "enabled": self.janitor_config.enabled,
            "schedule": self.janitor_config.schedule,
            "cleanup_paths": self.janitor_config.cleanup_paths,
            "file_patterns": self.janitor_config.file_patterns,
            "retention_hours": self.janitor_config.retention_hours,
            "dry_run": self.janitor_config.dry_run,
            "exclude_patterns": self.janitor_config.exclude_patterns,
            "log_deletions": self.janitor_config.log_deletions,
            "preserve_recent_files": self.janitor_config.preserve_recent_files
        }
        
        # Apply runtime overrides
        config.update(self._runtime_config)
        return config
    
    async def get_cleanup_preview(self) -> Dict[str, Any]:
        """
        Get preview of what would be cleaned up without actually deleting
        """
        logger.info("Generating cleanup preview")
        
        # Get effective config and temporarily enable dry run for preview
        effective_config = self.get_effective_config()
        original_dry_run = effective_config.get("dry_run", False)
        
        # Create a copy of effective config with dry run enabled
        preview_config = effective_config.copy()
        preview_config["dry_run"] = True
        
        try:
            # Run cleanup with preview config
            cleanup_stats = {
                "start_time": datetime.now().isoformat(),
                "files_deleted": 0,
                "bytes_freed": 0,
                "errors": [],
                "paths_processed": []
            }
            
            cutoff_time = datetime.now() - timedelta(hours=preview_config["retention_hours"])
            
            for cleanup_path in preview_config["cleanup_paths"]:
                full_path = self.base_path / cleanup_path
                if not full_path.exists():
                    logger.debug(f"Cleanup path does not exist: {full_path}")
                    continue
                    
                path_stats = await self._cleanup_path(full_path, cutoff_time, preview_config)
                cleanup_stats["files_deleted"] += path_stats["files_deleted"]
                cleanup_stats["bytes_freed"] += path_stats["bytes_freed"]
                cleanup_stats["errors"].extend(path_stats["errors"])
                cleanup_stats["paths_processed"].append({
                    "path": str(cleanup_path),
                    "files_deleted": path_stats["files_deleted"],
                    "bytes_freed": path_stats["bytes_freed"]
                })
                
            cleanup_stats["end_time"] = datetime.now().isoformat()
            cleanup_stats["duration_seconds"] = (
                datetime.fromisoformat(cleanup_stats["end_time"]) - 
                datetime.fromisoformat(cleanup_stats["start_time"])
            ).total_seconds()
            cleanup_stats["is_preview"] = True
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cleanup preview: {str(e)}", exc_info=True)
            return {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "files_deleted": 0,
                "bytes_freed": 0,
                "errors": [str(e)],
                "paths_processed": [],
                "is_preview": True,
                "duration_seconds": 0
            }

# Global janitor service instance
janitor_service = JanitorService()