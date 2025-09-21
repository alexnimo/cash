"""
File Janitor Service - Automated cleanup of generated files
"""
import os
import glob
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from croniter import croniter
import time

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
        
        cutoff_time = datetime.now() - timedelta(hours=self.janitor_config.keep_files_hours)
        
        try:
            # Process each cleanup path
            for cleanup_path in self.janitor_config.cleanup_paths:
                full_path = self.base_path / cleanup_path
                
                if not full_path.exists():
                    logger.debug(f"Cleanup path does not exist: {full_path}")
                    continue
                    
                path_stats = await self._cleanup_path(full_path, cutoff_time)
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
        
    async def _cleanup_path(self, path: Path, cutoff_time: datetime) -> Dict[str, Any]:
        """Clean up files in a specific path"""
        stats = {
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": []
        }
        
        try:
            # Process each file pattern
            for pattern in self.janitor_config.file_patterns:
                # Use glob to find matching files
                pattern_path = path / "**" / pattern
                
                for file_path in glob.glob(str(pattern_path), recursive=True):
                    file_path = Path(file_path)
                    
                    try:
                        # Check if file is old enough to delete
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            
                            if file_time < cutoff_time:
                                file_size = file_path.stat().st_size
                                
                                if self.janitor_config.dry_run:
                                    logger.info(f"[DRY RUN] Would delete: {file_path} ({file_size} bytes)")
                                else:
                                    file_path.unlink()
                                    logger.debug(f"Deleted file: {file_path} ({file_size} bytes)")
                                    
                                stats["files_deleted"] += 1
                                stats["bytes_freed"] += file_size
                                
                    except Exception as e:
                        error_msg = f"Error processing file {file_path}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
                        
        except Exception as e:
            error_msg = f"Error processing path {path}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            
        return stats
        
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
                "keep_files_hours": self.janitor_config.keep_files_hours
            }
        }
        
    async def manual_cleanup(self) -> Dict[str, Any]:
        """
        Trigger manual cleanup (useful for testing/admin)
        """
        logger.info("Manual cleanup triggered")
        return await self.cleanup_files()

# Global janitor service instance
janitor_service = JanitorService()