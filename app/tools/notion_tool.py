import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from notion_client import Client
from app.core.settings import get_settings
import requests
import json
import os

logger = logging.getLogger(__name__)

class NotionTool:
    def __init__(self):
        settings = get_settings()
        if not settings.notion.api_key:
            raise ValueError("Notion API key not found in settings")
            
        self.client = Client(auth=settings.notion.api_key)
        self.database_id = settings.notion.database_id
        self.stock_ticker_property = settings.notion.stock_ticker_property
        self.charts_property = settings.notion.charts_property
        self.ta_summary_property = settings.notion.ta_summary_property
        
        # Initialize database if needed
        self._init_database()
        
    def _init_database(self):
        """Initialize the Notion database with required properties."""
        try:
            # Check if database exists
            try:
                db = self.client.databases.retrieve(database_id=self.database_id)
                logger.info(f"Found existing database: {db['title'][0]['text']['content']}")
                
                # Get existing properties
                properties = db.get('properties', {})
                
                # Check and create required properties
                updates = {}
                
                # Stock Ticker property (title)
                if self.stock_ticker_property not in properties:
                    updates[self.stock_ticker_property] = {
                        "title": {}
                    }
                elif properties[self.stock_ticker_property]['type'] != 'title':
                    logger.error(f"{self.stock_ticker_property} must be a title property")
                    raise ValueError(f"{self.stock_ticker_property} must be a title property")
                
                # Charts property (files)
                if self.charts_property not in properties:
                    updates[self.charts_property] = {
                        "files": {}
                    }
                elif properties[self.charts_property]['type'] != 'files':
                    logger.error(f"{self.charts_property} must be a files property")
                    raise ValueError(f"{self.charts_property} must be a files property")
                
                # TA Summary property (rich text)
                if self.ta_summary_property not in properties:
                    updates[self.ta_summary_property] = {
                        "rich_text": {}
                    }
                elif properties[self.ta_summary_property]['type'] != 'rich_text':
                    logger.error(f"{self.ta_summary_property} must be a rich text property")
                    raise ValueError(f"{self.ta_summary_property} must be a rich text property")
                
                # Update database if needed
                if updates:
                    logger.info("Adding missing properties to database")
                    self.client.databases.update(
                        database_id=self.database_id,
                        properties=updates
                    )
                    logger.info("Database properties updated successfully")
                else:
                    logger.info("All required properties exist")
                
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                raise ValueError("Database not found or cannot be accessed. Please check the database ID and permissions.")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
            
    async def get_stock_page(self, ticker: str) -> Optional[Dict]:
        """Get existing page content for a stock."""
        try:
            # Convert ticker to uppercase for consistent comparison
            ticker = ticker.upper()
            response = self.client.databases.query(
                database_id=self.database_id,
                filter={
                    "property": self.stock_ticker_property,
                    "title": {
                        "equals": ticker
                    }
                }
            )
            
            if response["results"]:
                return response["results"][0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock page: {str(e)}")
            return None
            
    async def update_stock_entry(self, ticker: str, ta_summary: str) -> str:
        """Create or update a stock entry in Notion."""
        try:
            # Find existing page
            response = self.client.databases.query(
                database_id=self.database_id,
                filter={
                    "property": self.stock_ticker_property,
                    "title": {
                        "equals": ticker.upper()
                    }
                }
            )
            
            if response["results"]:
                # Update existing page
                page_id = response["results"][0]["id"]
                page = self.client.pages.retrieve(page_id)
                
                # Get existing properties
                properties = page["properties"]
                
                # Find the TA Summary property ID
                ta_summary_id = None
                for prop_id, prop in properties.items():
                    if prop["type"] == "rich_text" and prop_id == self.ta_summary_property:
                        ta_summary_id = prop_id
                        break
                
                if ta_summary_id:
                    # Update only the TA Summary property
                    self.client.pages.update(
                        page_id=page_id,
                        properties={
                            ta_summary_id: {
                                "rich_text": [{"text": {"content": ta_summary}}]
                            }
                        }
                    )
                else:
                    logger.error(f"TA Summary property not found for {ticker}")
                    raise ValueError(f"TA Summary property not found for {ticker}")
            else:
                # Create new page with all required properties
                response = self.client.pages.create(
                    parent={"database_id": self.database_id},
                    properties={
                        self.stock_ticker_property: {
                            "title": [{"text": {"content": ticker.upper()}}]
                        },
                        self.ta_summary_property: {
                            "rich_text": [{"text": {"content": ta_summary}}]
                        }
                    }
                )
                page_id = response["id"]
                
            return page_id
            
        except Exception as e:
            logger.error(f"Error updating stock entry: {str(e)}")
            raise
            
    async def get_channel_content(self, page_id: str, channel_name: str) -> Optional[str]:
        """Get existing content for a specific channel."""
        try:
            blocks = self.client.blocks.children.list(block_id=page_id)
            
            # Find Technical Analysis toggle
            for block in blocks["results"]:
                if (block["type"] == "toggle" and 
                    block["toggle"]["rich_text"][0]["text"]["content"] == "Technical Analysis"):
                    
                    # Find channel toggle
                    source_blocks = self.client.blocks.children.list(block_id=block["id"])
                    for src_block in source_blocks["results"]:
                        if (src_block["type"] == "toggle" and 
                            src_block["toggle"]["rich_text"][0]["text"]["content"] == channel_name):
                            
                            # Get channel content
                            content_blocks = self.client.blocks.children.list(block_id=src_block["id"])
                            for content_block in content_blocks["results"]:
                                if content_block["type"] == "paragraph":
                                    return content_block["paragraph"]["rich_text"][0]["text"]["content"]
                            
            return None
            
        except Exception as e:
            logger.error(f"Error getting channel content: {str(e)}")
            return None
            
    def _get_block_content(self, block: Dict) -> Optional[str]:
        """Safely get block content."""
        try:
            if block["type"] in ["toggle", "paragraph"]:
                rich_text = block[block["type"]].get("rich_text", [])
                if rich_text:
                    return rich_text[0]["text"]["content"]
            return None
        except Exception:
            return None
            
    async def update_technical_analysis(self, page_id: str, source: str, content: str, frame_paths: List[str] = None):
        """Update technical analysis section for a specific source."""
        try:
            # Find existing Technical Analysis toggle
            blocks = self.client.blocks.children.list(block_id=page_id)
            ta_block_id = None
            source_block_id = None
            
            # Find or create Technical Analysis toggle
            for block in blocks["results"]:
                block_content = self._get_block_content(block)
                if block_content == "Technical Analysis":
                    ta_block_id = block["id"]
                    break
                    
            if not ta_block_id:
                # Create Technical Analysis toggle
                response = self.client.blocks.children.append(
                    block_id=page_id,
                    children=[{
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": "Technical Analysis"}
                            }]
                        }
                    }]
                )
                ta_block_id = response["results"][0]["id"]
            
            # Find or create source toggle under Technical Analysis
            source_blocks = self.client.blocks.children.list(block_id=ta_block_id)
            for block in source_blocks["results"]:
                block_content = self._get_block_content(block)
                if block_content == source:
                    source_block_id = block["id"]
                    # Clear existing content
                    self.client.blocks.children.list(block_id=source_block_id)
                    break
            
            if not source_block_id:
                # Create new source toggle
                response = self.client.blocks.children.append(
                    block_id=ta_block_id,
                    children=[{
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": source}
                            }]
                        }
                    }]
                )
                source_block_id = response["results"][0]["id"]
            
            # Add content under source toggle
            if content:
                content_blocks = []
                
                # Split content into paragraphs
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        content_blocks.append({
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{
                                    "type": "text",
                                    "text": {"content": paragraph.strip()}
                                }]
                            }
                        })
                
                if content_blocks:
                    self.client.blocks.children.append(
                        block_id=source_block_id,
                        children=content_blocks
                    )
            
            # Add frames if provided
            if frame_paths:
                for frame_path in frame_paths:
                    await self.add_chart_to_page(page_id, frame_path)
            
            return "Successfully updated technical analysis"
            
        except Exception as e:
            logger.error(f"Error updating technical analysis: {str(e)}")
            return f"Error updating technical analysis: {str(e)}"
            
    async def get_all_tickers(self) -> List[str]:
        """Get all stock tickers from the Notion database."""
        try:
            response = self.client.databases.query(
                database_id=self.database_id
            )
            
            tickers = []
            for page in response["results"]:
                try:
                    ticker_prop = page["properties"].get(self.stock_ticker_property, {})
                    if ticker_prop and "title" in ticker_prop and ticker_prop["title"]:
                        # Store tickers in uppercase for consistent comparison
                        ticker = ticker_prop["title"][0]["plain_text"].upper()
                        tickers.append(ticker)
                except (KeyError, IndexError) as e:
                    logger.warning(f"Error extracting ticker from page: {str(e)}")
                    continue
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting tickers from Notion: {str(e)}", exc_info=True)
            raise

    async def _upload_to_freeimage(self, file_path: str) -> Optional[str]:
        """Upload a file to freeimage.host and return the public URL."""
        try:
            # Try getting API key directly from environment first
            api_key = os.getenv("FREEIMAGE_API_KEY")
            if not api_key:
                # Fallback to settings
                settings = get_settings()
                api_key = settings.freeimage.api_key
                
            logger.info(f"API key from env: {bool(os.getenv('FREEIMAGE_API_KEY'))}")
            logger.info(f"Settings has freeimage: {hasattr(get_settings(), 'freeimage')}")
            
            if not api_key:
                logger.error("No freeimage API key found in settings or environment")
                return None

            # Read file as binary
            with open(file_path, 'rb') as f:
                files = {
                    'source': (Path(file_path).name, f, 'image/jpeg')
                }
                data = {
                    'key': api_key,
                    'action': 'upload',
                    'format': 'json'
                }
                
                logger.info(f"Uploading {file_path} to freeimage.host...")
                response = requests.post(
                    'https://freeimage.host/api/1/upload',
                    files=files,
                    data=data
                )
                
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status_code') == 200:
                        image_url = data.get('image', {}).get('url')
                        logger.info(f"Successfully uploaded image, URL: {image_url}")
                        return image_url
                    else:
                        logger.error(f"Upload failed: {data.get('error', {}).get('message')}")
                else:
                    logger.error(f"Request failed with status {response.status_code}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error uploading to freeimage.host: {str(e)}")
            return None

    async def add_chart_to_page(self, page_id: str, chart_path: str, description: str = ""):
        """Add a chart image to the page's Charts property."""
        try:
            # Get current page
            page = self.client.pages.retrieve(page_id)
            
            # Get existing charts
            existing_charts = page["properties"].get(self.charts_property, {}).get("files", [])
            logger.info(f"Found {len(existing_charts)} existing charts")
            
            # Upload image to get public URL
            public_url = await self._upload_to_freeimage(chart_path)
            if not public_url:
                logger.error(f"Failed to upload {chart_path}")
                return
                
            logger.info(f"Got public URL: {public_url}")
            
            # Create file object for Notion
            new_chart = {
                "name": description or Path(chart_path).name,
                "type": "external",
                "external": {
                    "url": public_url
                }
            }
            
            # Add new chart to existing charts
            updated_charts = existing_charts + [new_chart]
            
            logger.info("Updating Notion page with new chart")
            # Update page with new charts
            self.client.pages.update(
                page_id=page_id,
                properties={
                    self.charts_property: {
                        "type": "files",
                        "files": updated_charts
                    }
                }
            )
            
            logger.info(f"Successfully added chart {chart_path} to page {page_id}")
            
        except Exception as e:
            logger.error(f"Error adding chart to page: {str(e)}")
            raise
