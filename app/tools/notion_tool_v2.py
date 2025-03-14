from llama_index.core import Response
from llama_index.core.tools import BaseTool, ToolMetadata
from notion_client import AsyncClient
from typing import Dict, List, Any, Optional, Literal
import os
import logging
import yaml
import uuid
from datetime import datetime
from pathlib import Path
from app.core.settings import get_settings
import requests
from pydantic import BaseModel, Field
import json
import traceback

logger = logging.getLogger(__name__)

def normalize_path_for_filesystem(path: str) -> str:
    """
    Normalize a path for cross-platform compatibility.
    Handles different path separators and formats.
    """
    # First try to use pathlib
    try:
        # This handles normal path conversion
        normalized_path = str(Path(path))
        if os.path.exists(normalized_path):
            return normalized_path
    except Exception:
        pass
        
    # Try alternate path format if needed
    alt_path = path
    if os.name == 'nt' and '/' in path:  # Windows
        alt_path = path.replace('/', '\\')
    elif os.name != 'nt' and '\\' in path:  # Unix/Linux
        alt_path = path.replace('\\', '/')
        
    # Special case for WSL paths when viewed from Windows or vice versa
    if path.startswith("/mnt/") and not os.path.exists(path) and os.name == 'nt':
        # Convert /mnt/c/... to C:\...
        drive_letter = path[5:6]
        # Fix the backslash in f-string issue
        alt_path = f"{drive_letter.upper()}:{path[6:]}".replace('/', '\\')
    elif path[1:3] == ":\\" and not os.path.exists(path) and os.name != 'nt':
        # Convert C:\... to /mnt/c/...
        drive_letter = path[0].lower()
        alt_path = f"/mnt/{drive_letter}{path[2:]}".replace('\\', '/')
        
    return alt_path

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class NotionToolSchema(BaseModel):
    """Schema for the Notion tool operations"""
    # The operation to perform
    operation: str = Field(
        description="Operation to perform (e.g., get_stock_page, create_or_update_stock_page)"
    )
    
    # Stock operations
    ticker: Optional[str] = Field(
        None, description="Stock ticker symbol (required for stock operations)"
    )
    
    # Page operations
    page_id: Optional[str] = Field(
        None, description="Notion page ID (required for page update operations)"
    )
    
    # Content fields
    content: Optional[Dict] = Field(
        None, description="Content data for updates"
    )
    
    # Additional parameters
    channel_name: Optional[str] = Field(
        None, description="Channel name for technical analysis"
    )
    
    image_path: Optional[str] = Field(
        None, description="Path to image for chart uploads"
    )
    
    description: Optional[str] = Field(
        None, description="Description for charts"
    )
    
    insights: Optional[Dict] = Field(
        None, description="Market insights data"
    )

class NotionAdvancedToolSpec(BaseTool):
    """Advanced Notion tool for interacting with Notion databases and pages."""
    
    def __init__(self, api_key=None, database_id=None):
        # Initialize parent class first
        super().__init__()
        
        # Load config
        config = load_config()
        settings = get_settings()
        
        # Check for required environment variables
        if not os.getenv('NOTION_API_KEY'):
            logger.error("NOTION_API_KEY environment variable not set")
        if not os.getenv('NOTION_DATABASE_ID'):
            logger.error("NOTION_DATABASE_ID environment variable not set")
        if not os.getenv('FREEIMAGE_API_KEY'):
            logger.error("FREEIMAGE_API_KEY environment variable not set")
            
        # Initialize Notion client
        self.notion = AsyncClient(auth=settings.notion.api_key if api_key is None else api_key)
        
        # Get database ID from settings
        self.database_id = settings.notion.database_id if database_id is None else database_id
        if not self.database_id:
            raise ValueError("Notion database ID is not configured")
            
        # Get property names from config.yaml
        notion_config = config.get('notion', {})
        self.properties = notion_config.get('properties', {})
        self.default_values = notion_config.get('default_values', {})
        
    def _validate_env_vars(self):
        """Validate required environment variables"""
        required_vars = ["NOTION_API_KEY", "NOTION_DATABASE_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata"""
        return ToolMetadata(
            name="notion_tool",
            description="""Tool for managing stock analysis in Notion.

EXAMPLES:

1. Check if stock exists:
   {"operation": "get_stock_page", "ticker": "TSLA"}

2. Create/update stock page:
   {
     "operation": "create_or_update_stock_page",
     "ticker": "TSLA",
     "content": {
       "summary": "Technical analysis...",
       "key_points": [...]
     }
   }

3. Add chart to page:
   {
     "operation": "add_chart_to_page",
     "page_id": "page_id_here",
     "image_path": "/path/to/image.jpg"
   }

4. Update technical analysis:
   {
     "operation": "update_technical_analysis",
     "page_id": "page_id_here", 
     "content": "Technical analysis text",
     "channel_name": "Channel Name"
   }

5. Upload chart image:
   {
     "operation": "upload_chart_image",
     "page_id": "page_id_here",
     "image_url": "https://example.com/image.jpg",
     "caption": "Chart caption"
   }

WORKFLOW STEPS:
1. First check if a stock page exists using get_stock_page
2. Then either update an existing page or create a new one
3. Finally, add any chart images to the page
""",
            fn_schema=NotionToolSchema
        )

    async def __call__(self, **kwargs):
        """Execute tool with given operation and parameters"""
        try:
            # Extract operation and parameters from the input
            operation = kwargs.get('operation')
            
            if not operation:
                raise ValueError("Missing required parameter: 'operation'")
                
            if operation == "get_stock_page":
                if "ticker" not in kwargs:
                    raise ValueError("Missing required parameter: 'ticker'")
                return await self.get_stock_page(kwargs["ticker"])
                
            elif operation == "create_or_update_stock_page":
                if "ticker" not in kwargs or "content" not in kwargs:
                    raise ValueError("Missing required parameters: 'ticker' and/or 'content'")
                return await self.create_or_update_stock_page(kwargs)
                
            elif operation == "update_technical_analysis":
                required = ["page_id", "content", "channel_name"]
                missing = [p for p in required if p not in kwargs]
                if missing:
                    raise ValueError(f"Missing required parameters: {missing}")
                return await self.update_technical_analysis(
                    kwargs["page_id"], kwargs["content"], kwargs["channel_name"]
                )
                
            elif operation == "add_chart_to_page":
                if "page_id" not in kwargs or "image_path" not in kwargs:
                    raise ValueError("Missing required parameters: 'page_id' and/or 'image_path'")
                return await self.add_chart_to_page(
                    kwargs["page_id"], 
                    kwargs["image_path"],
                    kwargs.get("description", "")
                )
                
            elif operation == "get_all_tickers":
                return await self.get_all_tickers()
                
            elif operation == "create_technical_analysis_section":
                if "page_id" not in kwargs or "content" not in kwargs:
                    raise ValueError("Missing required parameters: 'page_id' and/or 'content'")
                return await self.create_technical_analysis_section(kwargs)
                
            elif operation == "upload_chart_image":
                if "page_id" not in kwargs:
                    raise ValueError("Missing required parameter: 'page_id'")
                if "image_path" not in kwargs and "image_url" not in kwargs:
                    raise ValueError("Either 'image_path' or 'image_url' must be provided")
                return await self.upload_chart_image(kwargs)
                
            elif operation == "update_market_insights":
                if "page_id" not in kwargs or "insights" not in kwargs:
                    raise ValueError("Missing required parameters: 'page_id' and/or 'insights'")
                return await self.update_market_insights(kwargs)
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in NotionTool: {str(e)}")
            return {"status": "error", "message": str(e), "error_type": type(e).__name__}

    async def create_page(self, data: Dict) -> Dict:
        """Create a new page in Notion database"""
        try:
            response = await self.notion.pages.create(
                parent={"database_id": self.database_id},
                properties=data
            )
            return {"status": "success", "page_id": response["id"]}
        except Exception as e:
            logger.error(f"Error creating Notion page: {str(e)}")
            raise

    async def update_page(self, data: Dict) -> Dict:
        """Update an existing page in Notion database"""
        try:
            page_id = data["page_id"]
            properties = data["properties"]
            
            # Format properties according to Notion API schema
            formatted_properties = {}
            for key, value in properties.items():
                prop_config = self.properties.get(key)
                if not prop_config:
                    continue
                    
                notion_key = prop_config["name"]
                prop_type = prop_config["type"]
                
                if prop_type == "title" and isinstance(value, str):
                    formatted_properties[notion_key] = {
                        "title": [{"text": {"content": value}}]
                    }
                elif prop_type == "rich_text" and isinstance(value, str):
                    formatted_properties[notion_key] = {
                        "rich_text": [{"text": {"content": value}}]
                    }
                elif prop_type == "files" and isinstance(value, dict):
                    formatted_properties[notion_key] = value
            
            # Update the page
            return await self.notion.pages.update(
                page_id=page_id,
                properties=formatted_properties
            )
        except Exception as e:
            logger.error(f"Error updating Notion page: {str(e)}")
            raise

    async def query_database(self, params: Dict) -> Dict:
        """Query Notion database with given parameters"""
        try:
            # Extract database_id from params
            database_id = params.pop('database_id')
            
            # Remove null values from params to avoid validation errors
            cleaned_params = {k: v for k, v in params.items() if v is not None}
            
            # Ensure sorts is either undefined or a valid array
            if 'sorts' in cleaned_params and not cleaned_params['sorts']:
                del cleaned_params['sorts']
            
            response = await self.notion.databases.query(
                database_id=database_id,
                **cleaned_params
            )
            return response
        except Exception as e:
            logger.error(f"Error querying Notion database: {str(e)}")
            raise

    async def get_tracked_stocks(self) -> List[str]:
        """Get list of tracked stocks from Notion database"""
        try:
            # Removed filter on 'Status' as it is not defined in config.yaml
            response = await self.query_database({
                "database_id": self.database_id,
                "sorts": [
                    {
                        "property": self.properties.get("stock_ticker", {}).get("name"),
                        "direction": "ascending"
                    }
                ]
            })
            
            tracked_stocks = []
            for page in response.get('results', []):
                symbol = page.get('properties', {}).get(self.properties.get("stock_ticker", {}).get("name"), {}).get('title', [])
                if symbol and symbol[0].get('text', {}).get('content'):
                    tracked_stocks.append(symbol[0]['text']['content'])
            
            return tracked_stocks
        except Exception as e:
            logger.error(f"Error getting tracked stocks: {str(e)}")
            raise

    async def create_technical_analysis_section(self, data: Dict) -> Dict:
        """Create structured TA section in Notion page"""
        try:
            page_id = data["page_id"]
            content = data["content"]
            return await self.notion.blocks.children.append(
                page_id,
                children=[{
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Technical Analysis"}}]
                    }
                }] + self._create_analysis_blocks(content)
            )
        except Exception as e:
            logger.error(f"Error creating technical analysis section: {str(e)}")
            raise

    async def upload_chart_image(self, data: Dict) -> Dict:
        """Upload chart image to Notion page"""
        try:
            page_id = data["page_id"]
            image_url = data["image_url"]
            caption = data.get("caption", "Technical Analysis Chart")
            return await self.notion.blocks.children.append(
                page_id,
                children=[{
                    "object": "block",
                    "type": "image",
                    "image": {
                        "type": "external",
                        "external": {"url": image_url},
                        "caption": [{"text": {"content": caption}}]
                    }
                }]
            )
        except Exception as e:
            logger.error(f"Error uploading chart image: {str(e)}")
            raise

    async def update_market_insights(self, data: Dict) -> Dict:
        """Update market insights database section"""
        try:
            page_id = data["page_id"]
            insights = data["insights"]
            return await self.notion.blocks.children.append(
                page_id,
                children=[{
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Market Insights"}}]
                    }
                }] + self._create_analysis_blocks(insights)
            )
        except Exception as e:
            logger.error(f"Error updating market insights: {str(e)}")
            raise

    async def create_or_update_stock_page(self, data: Dict) -> Dict:
        """
        Create or update a stock page in the database.
        
        Args:
            data (Dict): Dictionary containing ticker and content.
            
        Returns:
            Dict: Page object if created or updated, error message if failed.
            
        Example:
            stock_page = await notion_tool.create_or_update_stock_page({
                "ticker": "AAPL",
                "content": {
                    "summary": "Apple Inc. is a technology company.",
                    "key_points": ["Point 1", "Point 2"]
                }
            })
            if stock_page["status"] == "success":
                # Page created or updated successfully
            else:
                # Error occurred
        """
        try:
            ticker = data["ticker"]
            content = data["content"]
            logger.info(f"Processing stock page for {ticker}")
            logger.info(f"Content received: {content}")
            
            # Extract frame paths
            frame_paths = content.get("frame_paths", [])
            if frame_paths:
                logger.info(f"Found frame paths for {ticker}: {frame_paths}")
                # Verify files exist
                valid_frames = []
                for path in frame_paths:
                    normalized_path = normalize_path_for_filesystem(path)
                    if not os.path.exists(normalized_path):
                        logger.error(f"Frame file does not exist: {path}")
                    else:
                        logger.info(f"Frame file exists: {path} (size: {os.path.getsize(normalized_path)} bytes)")
                        valid_frames.append(normalized_path)
                frame_paths = valid_frames

            # Search for existing page
            results = await self.notion.databases.query(
                database_id=self.database_id,
                filter={
                    "property": self.properties.get("stock_ticker", {}).get("name"),
                    "title": {"equals": ticker}
                }
            )

            if results["results"]:
                # Update existing page
                page = results["results"][0]
                page_id = page["id"]
                
                # First get the page to check which properties actually exist
                page_details = await self.notion.pages.retrieve(page_id=page_id)
                existing_properties = page_details.get("properties", {})
                valid_properties = {}
                
                # Property names
                ta_summary_name = self.properties.get("ta_summary", {}).get("name", "TA Summary")
                key_points_name = self.properties.get("key_points", {}).get("name", "Key Points") 
                date_name = self.properties.get("date", {}).get("name", "Date")
                source_name = self.properties.get("source", {}).get("name", "Source")
                
                # Only add properties that exist
                if ta_summary_name in existing_properties:
                    valid_properties[ta_summary_name] = {
                        "rich_text": [{"text": {"content": content.get("summary", "")[:2000]}}]
                    }
                
                if key_points_name in existing_properties:
                    key_points_text = "\n• " + "\n• ".join(content.get("key_points", []))
                    valid_properties[key_points_name] = {
                        "rich_text": [{"text": {"content": key_points_text[:2000]}}]
                    }
                
                if date_name in existing_properties:
                    valid_properties[date_name] = {
                        "date": {"start": datetime.now().isoformat()}
                    }
                
                if source_name in existing_properties:
                    valid_properties[source_name] = {
                        "select": {"name": "AI Analysis"}
                    }
                
                # Update properties if any valid ones exist
                if valid_properties:
                    # Update properties
                    await self.notion.pages.update(
                        page_id=page_id,
                        properties=valid_properties
                    )
                
                # Upload and attach frames if present
                if frame_paths:
                    logger.info(f"Processing {len(frame_paths)} frames for {ticker}")
                    for frame_path in frame_paths:
                        try:
                            # Upload to freeimage.host
                            image_url = await self._upload_to_freeimage(frame_path)
                            if image_url:
                                logger.info(f"Successfully uploaded frame to freeimage: {image_url}")
                                # Add to Notion page
                                await self.add_chart_to_page(page_id, frame_path, image_url)
                            else:
                                logger.error(f"Failed to upload frame to freeimage: {frame_path}")
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_path}: {str(e)}")
                
                # Also update the page content with blocks
                try:
                    # Instead of creating blocks directly, use the _update_page_with_ta_content method
                    # This will ensure that content is properly organized under Technical Analysis
                    await self._update_page_with_ta_content(
                        page_id=page_id,
                        summary=content.get("summary", ""),
                        key_points=content.get("key_points", []),
                        channel_name=content.get("channel_name", "Unknown Channel")
                    )
                    logger.info("Successfully updated page content using _update_page_with_ta_content")
                except Exception as block_error:
                    logger.error(f"Error adding content blocks to page: {str(block_error)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue even if block update fails
                
                return {"status": "success", "page_id": page_id}
            else:
                logger.info(f"No existing page found for ticker {ticker}")
                return {"status": "error", "message": f"No existing page found for ticker {ticker}"}
        except Exception as e:
            logger.error(f"Error creating/updating stock page: {str(e)}")
            raise

    async def create_stock_page(self, ticker: str, content: Dict) -> Dict:
        """Create a new stock page in the database"""
        try:
            properties = {
                "stock_ticker": ticker
            }
            
            # Handle content based on type
            if isinstance(content, str):
                # If content is a string, treat it as TA summary
                properties["ta_summary"] = content
            else:
                # If content is a dict, extract specific fields
                if content.get("technical_analysis"):
                    properties["ta_summary"] = content["technical_analysis"]
                if content.get("key_points"):
                    properties["key_points"] = content["key_points"]
            
            # Create the page
            response = await self.notion.pages.create(
                parent={"database_id": self.database_id},
                properties=properties
            )
            
            return {"status": "success", "page_id": response["id"]}
            
        except Exception as e:
            logger.error(f"Error creating stock page: {str(e)}")
            raise
            
    async def get_all_tickers(self) -> List[str]:
        """Get all ticker symbols from the Notion database."""
        try:
            # First get database schema to find the ticker property
            db = await self.notion.databases.retrieve(self.database_id)
            ticker_property = None
            for prop_id, prop in db.get('properties', {}).items():
                if prop.get('type') == 'title':
                    ticker_property = prop_id
                    break
                    
            if not ticker_property:
                logger.error("Could not find title property in database schema")
                return []
                
            # Query database with correct property ID
            query_result = await self.notion.databases.query(
                database_id=self.database_id,
                filter={
                    "property": ticker_property,
                    "title": {
                        "is_not_empty": True
                    }
                }
            )
            
            tracked_stocks = []
            for page in query_result.get('results', []):
                properties = page.get('properties', {})
                for prop_id, prop_value in properties.items():
                    if prop_id == ticker_property and prop_value.get('type') == 'title':
                        title_content = prop_value.get('title', [{}])[0].get('text', {}).get('content', '')
                        if title_content:
                            tracked_stocks.append(title_content)
            
            # Log as a properly formatted string
            if tracked_stocks:
                stocks_str = ", ".join(tracked_stocks)
                logger.info(f"Found {len(tracked_stocks)} tracked stocks: {stocks_str}")
            else:
                logger.info(f"Found {len(tracked_stocks)} tracked stocks")
                
            return tracked_stocks
            
        except Exception as e:
            logger.error(f"Error getting tickers from Notion: {str(e)}")
            raise
            
    async def update_technical_analysis(self, page_id: str, content: str, channel_name: str) -> Dict:
        """
        Update technical analysis content for a stock page.
        
        Args:
            page_id (str): ID of the stock page.
            content (str): Technical analysis content.
            channel_name (str): Name of the channel.
            
        Returns:
            Dict: Success message if updated, error message if failed.
            
        Example:
            result = await notion_tool.update_technical_analysis(
                page_id="page_id",
                content="Technical analysis content",
                channel_name="Channel Name"
            )
            if result["status"] == "success":
                # Technical analysis updated successfully
            else:
                # Error occurred
        """
        try:
            # Retrieve the page first to check what properties actually exist
            page = await self.notion.pages.retrieve(page_id=page_id)
            existing_properties = page.get("properties", {})
            
            # Debug message to show database properties
            logger.info(f"Database properties: {list(self.properties.keys())}")
            logger.info(f"Available properties in database: {self.properties}")
            logger.info(f"Existing page properties: {list(existing_properties.keys())}")
            
            # Process content
            if isinstance(content, str):
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"summary": content}
            else:
                content_dict = content
                
            summary = content_dict.get("summary", "")
            key_points = content_dict.get("key_points", [])
            
            # Format key points as bulleted list if it's a list
            if isinstance(key_points, list):
                key_points_text = "\n• " + "\n• ".join(key_points)
            else:
                key_points_text = key_points
                
            # Prepare properties to update - only include properties that exist
            update_props = {}
            
            # Map our property names to actual property names in the database
            highlights_name = self.properties.get("key_points", {}).get("name", "Highlights")  # Map key_points to Highlights
            update_date_name = self.properties.get("update_date", {}).get("name", "Update Date")  # Use Update Date instead of Date
            
            # Check which properties exist before adding them to update_props
            if highlights_name in existing_properties:
                update_props[highlights_name] = {
                    "rich_text": [{"text": {"content": key_points_text[:2000]}}]
                }
                logger.info(f"Will update '{highlights_name}' property with key points data")
            else:
                logger.info(f"Property '{highlights_name}' doesn't exist in the database")
                
            if update_date_name in existing_properties:
                update_props[update_date_name] = {
                    "date": {"start": datetime.now().isoformat()}
                }
                logger.info(f"Will update '{update_date_name}' property with current date")
            else:
                logger.info(f"Property '{update_date_name}' doesn't exist in the database")
            
            # Only attempt to update if we have properties to update
            if update_props:
                logger.info(f"Updating page {page_id} with properties: {list(update_props.keys())}")
                await self.notion.pages.update(
                    page_id=page_id,
                    properties=update_props
                )
                
                # Also update the page content with blocks
                try:
                    # Instead of creating blocks directly, use the _update_page_with_ta_content method
                    # This will ensure that content is properly organized under Technical Analysis
                    await self._update_page_with_ta_content(
                        page_id=page_id,
                        summary=summary,
                        key_points=key_points if isinstance(key_points, list) else [],
                        channel_name=channel_name or "Unknown Channel"
                    )
                    logger.info("Successfully updated page content using _update_page_with_ta_content")
                except Exception as block_error:
                    logger.error(f"Error adding content blocks to page: {str(block_error)}")
                    # Continue even if block update fails
                
                return {"status": "success"}
            else:
                logger.info(f"No valid properties to update for page {page_id}")
                return {"status": "warning", "message": f"No valid properties to update for page {page_id}"}
            
        except Exception as e:
            logger.error(f"Error updating technical analysis: {str(e)}")
            raise

    def _format_technical_content(self, content: str) -> List[Dict]:
        """Format technical analysis content with bullets and emphasis."""
        # Split content into lines
        lines = content.split('\n')
        formatted_blocks = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # Create bullet point for each line
            block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": []
                }
            }
            
            # Add emphasis to important phrases
            parts = line.split(':')
            if len(parts) > 1:
                # Make the part before colon bold
                block["bulleted_list_item"]["rich_text"].append({
                    "text": {"content": parts[0] + ": "},
                    "annotations": {"bold": True}
                })
                # Add the rest as normal text
                block["bulleted_list_item"]["rich_text"].append({
                    "text": {"content": ':'.join(parts[1:])}
                })
            else:
                block["bulleted_list_item"]["rich_text"].append({
                    "text": {"content": line}
                })
            
            formatted_blocks.append(block)
        
        return formatted_blocks

    def _extract_content_text(self, blocks: List[Dict]) -> str:
        """Extract text content from blocks for comparison."""
        text_parts = []
        for block in blocks:
            if block.get('type') == 'bulleted_list_item':
                rich_text = block.get('bulleted_list_item', {}).get('rich_text', [])
                text = ''.join(rt.get('text', {}).get('content', '') for rt in rich_text)
                text_parts.append(text)
        return '\n'.join(text_parts)

    async def _upload_to_freeimage(self, file_path: str) -> Optional[str]:
        """Upload a file to freeimage.host and return the public URL."""
        try:
            logger.info(f"Attempting to upload file to freeimage.host: {file_path}")
            
            # Verify file exists and is an image
            normalized_path = normalize_path_for_filesystem(file_path)
            if not os.path.exists(normalized_path):
                # Try alternate path format if the file doesn't exist
                alt_path = file_path
                if os.name == 'nt' and '/' in file_path:  # Windows
                    alt_path = file_path.replace('/', '\\')
                elif os.name != 'nt' and '\\' in file_path:  # Unix/Linux
                    alt_path = file_path.replace('\\', '/')
                
                if alt_path != file_path and os.path.exists(alt_path):
                    logger.info(f"Using alternate path format: {alt_path}")
                    file_path = alt_path
                else:
                    logger.error(f"File does not exist (tried both path formats): {file_path}")
                    return None
                
            # Get API key from environment
            api_key = os.getenv('FREEIMAGE_API_KEY')
            if not api_key:
                logger.error("FREEIMAGE_API_KEY environment variable not set")
                return None
                
            # Read file content
            with open(normalized_path, 'rb') as f:
                files = {'source': f}
                logger.info(f"File opened successfully, size: {os.path.getsize(normalized_path)} bytes")
                
                # Make POST request to freeimage.host
                response = requests.post(
                    'https://freeimage.host/api/1/upload',
                    params={'key': api_key},
                    files=files
                )
                logger.info(f"Freeimage.host response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    image_url = data.get('image', {}).get('url')
                    logger.info(f"Successfully uploaded image. URL: {image_url}")
                    return image_url
                else:
                    logger.error(f"Failed to upload image. Status code: {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error uploading to freeimage.host: {str(e)}")
            return None

    async def add_chart_to_page(self, page_id: str, chart_path: str, image_url: str = None) -> Dict:
        """
        Add a chart image to the page's Charts property.
        
        Args:
            page_id (str): ID of the page.
            chart_path (str): Path to the chart image.
            image_url (str, optional): URL of the chart image. Defaults to None.
            
        Returns:
            Dict: Success message if added, error message if failed.
            
        Example:
            result = await notion_tool.add_chart_to_page(
                page_id="page_id",
                chart_path="chart_path",
                image_url="image_url"
            )
            if result["status"] == "success":
                # Chart added successfully
            else:
                # Error occurred
        """
        try:
            logger.info(f"Adding chart to page {page_id}")
            logger.info(f"Chart path: {chart_path}")
            
            # Verify file exists
            normalized_path = normalize_path_for_filesystem(chart_path)
            if not os.path.exists(normalized_path):
                # Try alternate path format if the file doesn't exist
                alt_path = chart_path
                if os.name == 'nt' and '/' in chart_path:  # Windows
                    alt_path = chart_path.replace('/', '\\')
                elif os.name != 'nt' and '\\' in chart_path:  # Unix/Linux
                    alt_path = chart_path.replace('\\', '/')
                
                if alt_path != chart_path and os.path.exists(alt_path):
                    logger.info(f"Using alternate path format: {alt_path}")
                    normalized_path = alt_path
                else:
                    logger.error(f"Chart path does not exist (tried both path formats): {chart_path}")
                    return {"status": "error", "message": "Chart file not found"}
                
            logger.info(f"Image URL: {image_url}")
            
            if not image_url:
                # Upload to freeimage.host if URL not provided
                image_url = await self._upload_to_freeimage(normalized_path)
                
            if not image_url:
                logger.error("Failed to get image URL")
                return {"status": "error", "message": "Failed to upload image"}
                
            # Get the Charts property name from config
            charts_prop = self.properties.get("charts", {}).get("name")
            if not charts_prop:
                logger.error("Charts property not configured")
                return {"status": "error", "message": "Charts property not configured"}
                
            # Update the page's Charts property
            response = await self.notion.pages.update(
                page_id=page_id,
                properties={
                    charts_prop: {
                        "files": [{
                            "name": os.path.basename(normalized_path),
                            "type": "external",
                            "external": {"url": image_url}
                        }]
                    }
                }
            )
            logger.info(f"Successfully added chart to page")
            return {"status": "success", "response": response}
            
        except Exception as e:
            logger.error(f"Error adding chart to page: {str(e)}")
            raise

    async def get_stock_page(self, ticker: str) -> Optional[Dict]:
        """
        Check if a stock page exists in the Notion database.
        
        Args:
            ticker (str): The stock ticker symbol to check (e.g., "AAPL")
            
        Returns:
            Optional[Dict]: Page object if found, None if not found
            
        Example:
            stock_page = await notion_tool.get_stock_page("AAPL")
            if stock_page:
                # Page exists, use stock_page["id"] for further operations
            else:
                # Page does not exist, create it
        """
        try:
            # Search for existing page
            results = await self.notion.databases.query(
                database_id=self.database_id,
                filter={
                    "property": self.properties.get("stock_ticker", {}).get("name"),
                    "title": {"equals": ticker}
                }
            )

            if results["results"]:
                # Return existing page
                page = results["results"][0]
                page_id = page["id"]
                page_url = page.get("url", "")
                
                return {"status": "success", "page_id": page_id, "page_url": page_url, "ticker": ticker}
            else:
                logger.info(f"No existing page found for ticker {ticker}")
                return {"status": "error", "message": f"No existing page found for ticker {ticker}"}
        except Exception as e:
            logger.error(f"Error getting stock page: {str(e)}")
            raise

    def _create_analysis_blocks(self, content: Dict) -> List[Dict]:
        """Helper to format analysis content blocks"""
        blocks = []
        for k, v in content.items():
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "text": {"content": f" {k}: "},
                        "annotations": {"bold": True, "color": "blue"}
                    }, {
                        "text": {"content": str(v)}
                    }]
                }
            })
        return blocks

    def _create_page_properties(self, ticker: str, content: Dict) -> Dict:
        """Helper to create page properties for Notion API"""
        properties = {}
        
        # Add ticker property (title)
        ticker_prop = self.properties.get("stock_ticker", {}).get("name")
        if ticker_prop:
            properties[ticker_prop] = {"title": [{"text": {"content": ticker}}]}
        
        # Add TA summary if present
        ta_summary_prop = self.properties.get("ta_summary", {}).get("name")
        if ta_summary_prop and content.get("summary"):
            properties[ta_summary_prop] = {
                "rich_text": [{"text": {"content": content["summary"]}}]
            }
        
        # Add key points if present
        key_points_prop = self.properties.get("key_points", {}).get("name")
        if key_points_prop and content.get("key_points"):
            key_points_text = "\n• " + "\n• ".join(content["key_points"])
            properties[key_points_prop] = {
                "rich_text": [{"text": {"content": key_points_text}}]
            }
        
        # Add last updated date
        last_updated_prop = self.properties.get("last_updated", {}).get("name")
        if last_updated_prop:
            properties[last_updated_prop] = {
                "date": {"start": datetime.now().isoformat()}
            }
        
        return properties

    @staticmethod
    def example_usage():
        """
        Demonstrates example usage of the Notion tool.
        This is for documentation purposes only and doesn't actually run the operations.
        """
        # Example 1: Check if stock exists
        get_stock_example = {
            "operation": "get_stock_page",
            "ticker": "TSLA"
        }
        
        # Example 2: Create/update stock page
        create_stock_example = {
            "operation": "create_or_update_stock_page",
            "ticker": "TSLA",
            "content": {
                "summary": "Tesla showing strong support at current levels",
                "key_points": [
                    "Support level: $180",
                    "Resistance: $220",
                    "Volume indicates accumulation"
                ]
            }
        }
        
        # Example 3: Add chart to page
        chart_example = {
            "operation": "add_chart_to_page",
            "page_id": "page_12345",
            "image_path": "/path/to/chart.png",
            "description": "Tesla 4-hour price chart with volume"
        }
        
        return {
            "get_stock": get_stock_example,
            "create_stock": create_stock_example,
            "add_chart": chart_example
        }

class NotionToolParams(BaseModel):
    """Parameters for the Notion Tool"""
    operation: str = Field(
        ..., 
        description="The operation to perform on the Notion database",
        enum=["get_stock_page", "create_or_update_stock_page", "update_technical_analysis", "add_chart_to_page", "get_all_tickers"]
    )
    ticker: Optional[str] = Field(None, description="Stock ticker symbol (required for stock-related operations)")
    content: Optional[Any] = Field(None, description="Content for page creation or update")
    page_id: Optional[str] = Field(None, description="Notion page ID (required for update operations)")
    channel_name: Optional[str] = Field(None, description="Optional channel name for technical analysis updates")
    image_path: Optional[str] = Field(None, description="Path to image file for add_chart_to_page operation")
    description: Optional[str] = Field(None, description="Optional description for chart images")

class NotionTool(BaseTool):
    """Tool for interacting with Notion database to create and update stock analysis pages."""
    
    def __init__(self, notion_api_key: Optional[str] = None, notion_database_id: Optional[str] = None):
        """Initialize NotionTool with API key and database ID."""
        
        # Load API keys
        self.api_key = notion_api_key or os.environ.get('NOTION_API_KEY')
        if not self.api_key:
            raise ValueError("Missing NOTION_API_KEY environment variable")
            
        self.database_id = notion_database_id or os.environ.get('NOTION_DATABASE_ID')
        if not self.database_id:
            raise ValueError("Missing NOTION_DATABASE_ID environment variable")
            
        # Initialize Notion client
        self.notion = AsyncClient(auth=self.api_key)
        
        # Load properties config
        try:
            settings = get_settings()
            config_path = getattr(settings.notion, 'config_path', 'app/config/notion_config.yaml') 
            # Add more detailed logging for debugging
            logger.info(f"Using Notion config path: {config_path}")
            
            # Check if file exists before trying to open it
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.properties = config.get('properties', {})
                    logger.info(f"Loaded properties from config: {list(self.properties.keys())}")
            else:
                logger.warning(f"Config file not found at path: {config_path}. Using default properties.")
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        except Exception as e:
            logger.error(f"Error loading Notion config: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Fallback to default properties
            self.properties = {
                "stock_ticker": {"name": "Stock Ticker", "type": "title"},
                "ta_summary": {"name": "TA Summary", "type": "rich_text"},
                "key_points": {"name": "Key Points", "type": "rich_text"},
                "charts": {"name": "Charts", "type": "files"},
                "date": {"name": "Date", "type": "date"},
                "source": {"name": "Source", "type": "select"}
            }
            logger.info("Using default properties due to config loading error")
        
        # Call parent init without passing metadata - we'll implement the method instead
        super().__init__()
        
        logger.info(f"NotionTool initialized with database ID: {self.database_id[:5]}...")
    
    @property
    def metadata(self) -> ToolMetadata:
        """Implement the metadata property required by BaseTool"""
        return ToolMetadata(
            name="notion_tool",
            description="""Tool for creating and updating stock analysis pages in Notion.
            
Operations:
- get_stock_page: Get a stock page by ticker symbol
- create_or_update_stock_page: Create or update a stock page with content
- update_technical_analysis: Update technical analysis section of a page
- add_chart_to_page: Add a chart image to a page
- get_all_tickers: Get all stock tickers from the database
            """
        )
        
    async def get_all_tickers(self) -> list:
        """Get all stock tickers from the Notion database."""
        logger.debug("NotionTool.get_all_tickers called")
        try:
            tickers = []
            has_more = True
            start_cursor = None
            
            while has_more:
                query_params = {
                    "database_id": self.database_id,
                    "page_size": 100  # Maximum allowed by Notion API
                }
                
                if start_cursor:
                    query_params["start_cursor"] = start_cursor
                
                response = await self.notion.databases.query(**query_params)
                
                # Extract ticker from each page
                for page in response.get("results", []):
                    ticker_property = page.get("properties", {}).get(
                        self.properties.get("stock_ticker", {}).get("name"), {})
                    
                    if ticker_property.get("title"):
                        title_content = ticker_property["title"][0]["text"]["content"] if ticker_property["title"] else ""
                        if title_content:
                            tickers.append(title_content)
                            
                # Check if there are more pages
                has_more = response.get("has_more", False)
                if has_more:
                    start_cursor = response.get("next_cursor")
            
            return tickers
            
        except Exception as e:
            error_msg = f"Error retrieving tickers: {str(e)}"
            logger.error(error_msg)
            return []
    
    async def get_stock_page(self, ticker: str) -> dict:
        """Get a stock page by ticker symbol."""
        logger.debug(f"NotionTool.get_stock_page called with ticker: {ticker}")
        try:
            # Query Notion database for page with matching ticker
            query_params = {
                "database_id": self.database_id,
                "filter": {
                    "property": self.properties.get("stock_ticker", {}).get("name"),
                    "title": {"equals": ticker.upper()}
                }
            }
            
            response = await self.notion.databases.query(**query_params)
            
            if not response.get("results"):
                return {
                    "status": "error", 
                    "message": f"No page found for ticker: {ticker}"
                }
                
            # Return the first matching page
            page = response["results"][0]
            page_id = page["id"]
            page_url = page.get("url", "")
            
            return {
                "status": "success",
                "page_id": page_id,
                "page_url": page_url,
                "ticker": ticker
            }
            
        except Exception as e:
            error_msg = f"Error retrieving stock page for {ticker}: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    async def create_or_update_stock_page(self, ticker: str, content: Any) -> dict:
        """Create or update a stock page with content."""
        logger.debug(f"NotionTool.create_or_update_stock_page called for ticker: {ticker}")
        try:
            # Process content
            if isinstance(content, str):
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"summary": content}
            else:
                content_dict = content
                
            summary = content_dict.get("summary", "")
            key_points = content_dict.get("key_points", [])
            
            # Format key points as bulleted list if it's a list
            if isinstance(key_points, list):
                key_points_text = "\n".join([f"• {point}" for point in key_points])
            else:
                key_points_text = key_points
            
            # Check if page exists
            page = await self.get_stock_page(ticker)
            
            if page:
                # Update existing page
                page_id = page.get("id")
                logger.info(f"Found existing page for {ticker}, updating...")
                
                # Get existing properties to check what's available
                existing_page = await self.notion.pages.retrieve(page_id=page_id)
                existing_properties = existing_page.get("properties", {})
                
                # Only update properties that exist
                valid_properties = {}
                
                # Property names
                highlights_name = self.properties.get("key_points", {}).get("name", "Highlights")
                update_date_name = self.properties.get("update_date", {}).get("name", "Update Date")
                
                # Only add properties that exist
                if highlights_name in existing_properties:
                    valid_properties[highlights_name] = {
                        "rich_text": [{"text": {"content": key_points_text[:2000]}}]
                    }
                
                if update_date_name in existing_properties:
                    valid_properties[update_date_name] = {
                        "date": {"start": datetime.now().isoformat()}
                    }
                
                # Update the page with valid properties
                if valid_properties:
                    # Update properties
                    await self.notion.pages.update(
                        page_id=page_id,
                        properties=valid_properties
                    )
                    
                    # Update page content
                    try:
                        # Use the _update_page_with_ta_content method for consistency with the toggle functionality
                        await self._update_page_with_ta_content(
                            page_id=page_id,
                            summary=summary,
                            key_points=key_points if isinstance(key_points, list) else [],
                            channel_name=content_dict.get("channel_name", "Unknown Channel")
                        )
                        logger.info("Successfully updated page content using _update_page_with_ta_content")
                    except Exception as block_error:
                        logger.error(f"Error adding content blocks to page: {str(block_error)}")
                        # Continue even if block update fails
                
                return {
                    "status": "success",
                    "message": f"Updated page for {ticker}",
                    "page_id": page_id
                }
                
            # If page doesn't exist, create it
            else:
                # Create new page with properties
                properties = {
                    self.properties.get("stock_ticker", {}).get("name", "Stock Ticker"): {
                        "title": [{"text": {"content": ticker.upper()}}]
                    },
                    self.properties.get("key_points", {}).get("name", "Highlights"): {
                        "rich_text": [{"text": {"content": key_points_text[:2000]}}]
                    },
                    self.properties.get("update_date", {}).get("name", "Update Date"): {
                        "date": {"start": datetime.now().isoformat()}
                    }
                }
                
                response = await self.notion.pages.create(
                    parent={"database_id": self.database_id},
                    properties=properties
                )
                
                page_id = response["id"]
                
                # Also update the page content with Technical Analysis toggle
                try:
                    await self._update_page_with_ta_content(
                        page_id=page_id,
                        summary=summary,
                        key_points=key_points if isinstance(key_points, list) else [],
                        channel_name=content_dict.get("channel_name", "Unknown Channel")
                    )
                    logger.info("Successfully added content to new page using _update_page_with_ta_content")
                except Exception as e:
                    logger.error(f"Error adding content to new page: {str(e)}")
                    # Continue even if block update fails
                
                return {
                    "status": "success",
                    "message": f"Created new page for {ticker}",
                    "page_id": page_id
                }
                
        except Exception as e:
            error_msg = f"Error creating/updating stock page for {ticker}: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    async def update_technical_analysis(self, page_id: str, content: Any, channel_name: str = None) -> dict:
        """Update technical analysis section of a page."""
        logger.debug(f"NotionTool.update_technical_analysis called for page_id: {page_id}")
        try:
            # Retrieve the page first to check what properties actually exist
            page = await self.notion.pages.retrieve(page_id=page_id)
            existing_properties = page.get("properties", {})
            
            # Debug message to show database properties
            logger.info(f"Database properties: {list(self.properties.keys())}")
            logger.info(f"Available properties in database: {self.properties}")
            logger.info(f"Existing page properties: {list(existing_properties.keys())}")
            
            # Process content
            if isinstance(content, str):
                try:
                    content_dict = json.loads(content)
                except json.JSONDecodeError:
                    content_dict = {"summary": content}
            else:
                content_dict = content
                
            summary = content_dict.get("summary", "")
            key_points = content_dict.get("key_points", [])
            
            # Format key points as bulleted list if it's a list
            if isinstance(key_points, list):
                key_points_text = "\n".join([f"• {point}" for point in key_points])
            else:
                key_points_text = key_points
                
            # Prepare properties to update - only include properties that exist
            update_props = {}
            
            # Map property names to actual property names in the database using config
            highlights_name = self.properties.get("key_points", {}).get("name", "Highlights")  # Map key_points to Highlights
            update_date_name = self.properties.get("update_date", {}).get("name", "Update Date")  # Use Update Date instead of Date
            
            # Check which properties exist before adding them to update_props
            if highlights_name in existing_properties:
                update_props[highlights_name] = {
                    "rich_text": [{"text": {"content": key_points_text[:2000]}}]
                }
                logger.info(f"Will update '{highlights_name}' property with key points data")
            else:
                logger.info(f"Property '{highlights_name}' doesn't exist in the database")
                
            if update_date_name in existing_properties:
                update_props[update_date_name] = {
                    "date": {"start": datetime.now().isoformat()}
                }
                logger.info(f"Will update '{update_date_name}' property with current date")
            else:
                logger.info(f"Property '{update_date_name}' doesn't exist in the database")
            
            # Only attempt to update if we have properties to update
            if update_props:
                logger.info(f"Updating page {page_id} with properties: {list(update_props.keys())}")
                await self.notion.pages.update(
                    page_id=page_id,
                    properties=update_props
                )
                
                # Also update the page content with blocks
                try:
                    # Instead of creating blocks directly, use the _update_page_with_ta_content method
                    # This will ensure that content is properly organized under Technical Analysis
                    await self._update_page_with_ta_content(
                        page_id=page_id,
                        summary=summary,
                        key_points=key_points if isinstance(key_points, list) else [],
                        channel_name=channel_name or "Unknown Channel"
                    )
                    logger.info("Successfully updated page content using _update_page_with_ta_content")
                except Exception as block_error:
                    logger.error(f"Error adding content blocks to page: {str(block_error)}")
                    # Continue even if block update fails
                
                return {
                    "status": "success",
                    "message": f"Updated technical analysis for page {page_id}",
                    "page_id": page_id
                }
            else:
                logger.info(f"No valid properties to update for page {page_id}")
                return {
                    "status": "warning",
                    "message": f"No valid properties to update for page {page_id}",
                    "page_id": page_id
                }
            
        except Exception as e:
            error_msg = f"Error updating technical analysis for page {page_id}: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    async def add_chart_to_page(self, page_id: str, image_path: str, description: str = None) -> dict:
        """Add a chart image to a page."""
        logger.debug(f"NotionTool.add_chart_to_page called for page_id: {page_id}")
        try:
            # Check if file is local or remote URL
            if os.path.exists(image_path):
                # Upload local file to external hosting, since Notion API requires a URL
                try:
                    # Load API key for image hosting
                    api_key = os.environ.get('FREEIMAGE_API_KEY')
                    if not api_key:
                        raise ValueError("Missing FREEIMAGE_API_KEY environment variable")
                    
                    # Upload image to hosting service
                    with open(image_path, 'rb') as img_file:
                        response = requests.post(
                            'https://freeimage.host/api/1/upload',
                            files={'source': img_file},
                            data={'key': api_key}
                        )
                    
                    data = response.json()
                    if data.get('status_code') != 200:
                        raise ValueError(f"Image upload failed: {data.get('status_txt')}")
                    
                    image_url = data['image']['url']
                except Exception as e:
                    error_msg = f"Error uploading image: {str(e)}"
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
            else:
                # Assume it's already a URL
                image_url = image_path
            
            # Get existing files first
            page_response = await self.notion.pages.retrieve(page_id=page_id)
            existing_charts = page_response.get('properties', {}).get(
                self.properties.get("charts", {}).get("name", "Charts"), {}).get('files', [])
            
            # Prepare the file for addition
            new_file = {
                "name": description or f"Chart {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "type": "external",
                "external": {"url": image_url}
            }
            
            # Combine existing and new files
            updated_files = existing_charts + [new_file]
            
            # Update page with new file list
            await self.notion.pages.update(
                page_id=page_id,
                properties={
                    self.properties.get("charts", {}).get("name", "Charts"): {
                        "files": updated_files
                    }
                }
            )
            
            return {
                "status": "success",
                "message": f"Added chart to page {page_id}",
                "page_id": page_id,
                "image_url": image_url
            }
            
        except Exception as e:
            error_msg = f"Error adding chart to page {page_id}: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
    async def _arun(self, **kwargs) -> Any:
        """
        Async entry point for LlamaIndex tool protocol.
        
        This method is required for compatibility with the LlamaIndex ReAct agent
        and directly routes to the appropriate operation.
        """
        try:
            logger.info(f"NotionTool._arun called with: {kwargs}")
            
            # Direct execution based on operation field
            operation = kwargs.get('operation')
            
            if not operation:
                error_msg = "Missing required parameter: 'operation'"
                logger.error(error_msg)
                return json.dumps({"status": "error", "message": error_msg})
                
            # Map operations directly without requiring 'input' wrapper
            if operation == "get_stock_page":
                ticker = kwargs.get('ticker')
                if not ticker:
                    return json.dumps({"status": "error", "message": "Missing required parameter: 'ticker'"})
                result = await self.get_stock_page(ticker)
            elif operation == "create_or_update_stock_page":
                ticker = kwargs.get('ticker')
                content = kwargs.get('content')
                if not ticker or not content:
                    return json.dumps({"status": "error", "message": "Missing required parameters for create_or_update_stock_page"})
                result = await self.create_or_update_stock_page(ticker, content)
            elif operation == "update_technical_analysis":
                page_id = kwargs.get('page_id')
                content = kwargs.get('content')
                channel_name = kwargs.get('channel_name', None)
                if not page_id or not content:
                    return json.dumps({"status": "error", "message": "Missing required parameters for update_technical_analysis"})
                result = await self.update_technical_analysis(
                    page_id, content, channel_name
                )
            elif operation == "add_chart_to_page":
                page_id = kwargs.get('page_id')
                image_path = kwargs.get('image_path')
                description = kwargs.get('description', '')
                if not page_id or not image_path:
                    return json.dumps({"status": "error", "message": "Missing required parameters for add_chart_to_page"})
                result = await self.add_chart_to_page(
                    page_id, 
                    image_path,
                    description
                )
            elif operation == "get_all_tickers":
                result = await self.get_all_tickers()
                return json.dumps({"status": "success", "tickers": result})
            else:
                return json.dumps({"status": "error", "message": f"Unknown operation: {operation}"})
            
            # Convert result to JSON string if it's a dict
            if isinstance(result, dict):
                return json.dumps(result)
            return result
            
        except Exception as e:
            error_message = f"Error in NotionTool: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            return json.dumps({"status": "error", "message": error_message})
            
    async def arun(self, *args, **kwargs) -> Any:
        """
        Public async entry point that delegates to _arun.
        This method is called by the agent framework.
        """
        # If first arg is a dict, use it as kwargs
        if args and isinstance(args[0], dict):
            kwargs.update(args[0])
            
        return await self._arun(**kwargs)
            
    async def __call__(self, operation: Optional[str] = None, ticker: Optional[str] = None, 
                     content: Optional[Any] = None, page_id: Optional[str] = None, 
                     channel_name: Optional[str] = None, image_path: Optional[str] = None, 
                     description: Optional[str] = None, input: Optional[Any] = None, **kwargs) -> Any:
        """
        Standard implementation for LlamaIndex tool interface.
        This gets called by both types of agent frameworks:
        1. ReAct agent with parameter dictionaries
        2. Direct function calling with named parameters
        
        Returns:
            String result for the agent
        """
        try:
            # Add detailed logging for debugging
            logger.info("=" * 40)
            logger.info(f"NotionTool.__call__ called with:")
            logger.info(f"  operation: {operation}")
            logger.info(f"  ticker: {ticker}")
            logger.info(f"  content: {content}")
            logger.info(f"  page_id: {page_id}")
            logger.info(f"  channel_name: {channel_name}")
            logger.info(f"  image_path: {image_path}")
            logger.info(f"  description: {description}")
            logger.info(f"  input: {input}")
            logger.info(f"  kwargs: {kwargs}")
            
            # Handle ReAct agent input format - look for nested input field
            if input is not None and isinstance(input, dict):
                logger.info(f"Found input parameter, extracting operation parameters from it")
                operation = input.get('operation', operation)
                ticker = input.get('ticker', ticker)
                content = input.get('content', content)
                page_id = input.get('page_id', page_id)
                channel_name = input.get('channel_name', channel_name)
                image_path = input.get('image_path', image_path)
                description = input.get('description', description)
            
            if not operation:
                logger.error("No operation specified in NotionTool call")
                return "Error: No operation specified"
                
            logger.info(f"Executing operation: {operation}")
            
            # Initialize params dictionary
            params = {}
            
            # Process direct parameters if provided
            if operation:
                params['operation'] = operation
            if ticker:
                params['ticker'] = ticker
            if content is not None:
                params['content'] = content
            if page_id:
                params['page_id'] = page_id
            if channel_name:
                params['channel_name'] = channel_name
            if image_path:
                params['image_path'] = image_path
            if description:
                params['description'] = description
            
            # Handle various input formats
            if isinstance(input, dict):
                params.update(input)
            elif isinstance(input, str) and input.strip():
                try:
                    # Try to parse as JSON
                    input_dict = json.loads(input)
                    params.update(input_dict)
                except json.JSONDecodeError:
                    # If not JSON and no operation is set, assume get_stock_page
                    if 'operation' not in params:
                        params['operation'] = 'get_stock_page'
                    if 'ticker' not in params:
                        params['ticker'] = input
            
            # Handle special parameters wrappers
            if 'data' in params and isinstance(params['data'], dict):
                data_params = params.pop('data')
                params.update(data_params)
            
            # Also handle Action Input wrapper (common in ReAct output)
            if 'Action Input' in params:
                action_input = params.pop('Action Input')
                if isinstance(action_input, dict):
                    params.update(action_input)
                elif isinstance(action_input, str):
                    try:
                        action_dict = json.loads(action_input)
                        params.update(action_dict)
                    except json.JSONDecodeError:
                        # If parsing fails, treat as ticker for get_stock_page
                        if 'operation' not in params:
                            params['operation'] = 'get_stock_page'
                        if 'ticker' not in params:
                            params['ticker'] = action_input
            
            # Check for required parameters
            if not params.get('operation'):
                error = "Missing required parameter: 'operation'"
                logger.error(error)
                return json.dumps({"status": "error", "message": error})
            
            logger.debug(f"Processed params for NotionTool: {params}")
            
            # Route to the appropriate operation
            operation = params.get('operation')
            if operation == "get_stock_page" and params.get('ticker'):
                result = await self.get_stock_page(params['ticker'])
            elif operation == "create_or_update_stock_page" and params.get('ticker') and params.get('content'):
                result = await self.create_or_update_stock_page(params['ticker'], params['content'])
            elif operation == "update_technical_analysis" and params.get('page_id') and params.get('content'):
                result = await self.update_technical_analysis(
                    params['page_id'], params['content'], params.get('channel_name'))
            elif operation == "add_chart_to_page" and params.get('page_id') and params.get('image_path'):
                result = await self.add_chart_to_page(
                    params['page_id'], 
                    params['image_path'],
                    params.get('description'))
            elif operation == "get_all_tickers":
                result = await self.get_all_tickers()
            else:
                error_msg = f"Invalid operation or missing required parameters: {operation}"
                logger.error(error_msg)
                return json.dumps({"status": "error", "message": error_msg})
            
            # Return result as string
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            return str(result)
            
        except Exception as e:
            error_message = f"Error in NotionTool: {str(e)}"
            logger.error(error_message, exc_info=True)
            return json.dumps({"status": "error", "message": error_message})

    def _enhance_text_formatting(self, text):
        """Format text with rich text features like bold and color for key terms"""
        # Simple pattern matching for text enhancement
        if not text:
            return [{"text": {"content": ""}}]
            
        # Check for dollar amounts and percentages to make bold
        import re
        
        parts = []
        
        # Look for patterns like $123.45, +10%, -5%, keywords like support/resistance
        patterns = [
            # Price patterns
            (r'\$\d+\.?\d*', {"bold": True, "color": "blue"}),
            # Percentage patterns
            (r'\+\d+\.?\d*%', {"bold": True, "color": "green"}),
            (r'\-\d+\.?\d*%', {"bold": True, "color": "red"}),
            # Technical terms
            (r'\b(support|resistance|breakout|breakdown|trend|bullish|bearish)\b', {"bold": True}),
            # Emphasis on key metrics
            (r'\b(RSI|MACD|EMA|SMA|volume)\b', {"bold": True, "color": "purple"})
        ]
        
        # Find all matches
        matches = []
        for pattern, formatting in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((match.start(), match.end(), match.group(), formatting))
        
        # Sort matches by position
        matches.sort(key=lambda x: x[0])
        
        # Process the text with matches
        current_pos = 0
        for start, end, matched_text, formatting in matches:
            # Add text before match
            if start > current_pos:
                parts.append({"text": {"content": text[current_pos:start]}})
            
            # Add formatted match
            parts.append({"text": {"content": matched_text}, "annotations": formatting})
            current_pos = end
        
        # Add remaining text
        if current_pos < len(text):
            parts.append({"text": {"content": text[current_pos:]}})
        
        # If no patterns matched, return the original text
        if not parts:
            parts = [{"text": {"content": text}}]
            
        return parts

    async def _update_page_with_ta_content(self, page_id: str, summary: str, key_points: List[str], channel_name: str):
        """
        Update the page content with technical analysis blocks.
        Creates a nested structure with:
        - Technical Analysis (main toggle - heading_1 style)
          - Channel Name (channel toggle - heading_2 style)
            - Channel content (formatted text)
        """
        logger.info(f"Updating page {page_id} with technical analysis content for channel '{channel_name}'")
        
        # First check if a Technical Analysis toggle already exists
        blocks_response = await self.notion.blocks.children.list(block_id=page_id)
        tech_analysis_block_id = None
        
        # Iterate through blocks to find an existing Technical Analysis toggle
        for block in blocks_response.get("results", []):
            logger.info(f"Found block of type: {block.get('type')}")
            if block.get("type") == "toggle":
                toggle_text = block.get("toggle", {}).get("rich_text", [])
                for text_item in toggle_text:
                    content = text_item.get("text", {}).get("content", "")
                    logger.info(f"Checking toggle text content: '{content}'")
                    if content == "Technical Analysis":
                        tech_analysis_block_id = block.get("id")
                        logger.info(f"Found existing Technical Analysis toggle with ID: {tech_analysis_block_id}")
                        break
            if tech_analysis_block_id:
                break
        
        # If Technical Analysis section doesn't exist, create it
        if not tech_analysis_block_id:
            logger.info("Creating new Technical Analysis section")
            ta_response = await self.notion.blocks.children.append(
                block_id=page_id,
                children=[{
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Technical Analysis"},
                                "annotations": {"bold": True}
                            }
                        ]
                    }
                }]
            )
            
            # Get the ID of the newly created Technical Analysis toggle
            if ta_response.get("results") and len(ta_response.get("results", [])) > 0:
                tech_analysis_block_id = ta_response.get("results", [])[0].get("id")
                logger.info(f"Created Technical Analysis toggle with ID: {tech_analysis_block_id}")
            else:
                logger.error("Failed to create Technical Analysis toggle")
                return
        
        # Now that we have a Technical Analysis toggle (either existing or new),
        # check if this channel already exists as a child
        if tech_analysis_block_id:
            # List all children of the Technical Analysis toggle
            children_response = await self.notion.blocks.children.list(block_id=tech_analysis_block_id)
            ta_children = children_response.get("results", [])
            
            # Check for an existing channel section
            channel_block_id = None
            for child in ta_children:
                if child.get("type") == "toggle":
                    toggle_text = child.get("toggle", {}).get("rich_text", [])
                    for text_item in toggle_text:
                        if text_item.get("text", {}).get("content") == channel_name:
                            channel_block_id = child.get("id")
                            logger.info(f"Found existing channel section '{channel_name}' with ID: {channel_block_id}")
                            break
                if channel_block_id:
                    break
            
            # If channel section exists, delete it to replace it
            if channel_block_id:
                logger.info(f"Deleting existing channel section '{channel_name}'")
                await self.notion.blocks.delete(block_id=channel_block_id)
            
            # Create a new channel section toggle
            logger.info(f"Creating new toggle for channel '{channel_name}'")
            try:
                # Make sure we're creating a toggle block for the channel, not a heading
                channel_response = await self.notion.blocks.children.append(
                    block_id=tech_analysis_block_id,
                    children=[{
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": channel_name},
                                    "annotations": {"bold": True, "color": "blue"}
                                }
                            ]
                        }
                    }]
                )
                
                logger.info(f"Channel response structure: {str(channel_response)[:200]}...")
                
                # Get the ID of the newly created channel toggle
                new_channel_id = None
                if "results" in channel_response and len(channel_response["results"]) > 0:
                    new_channel_id = channel_response["results"][0].get("id")
                    logger.info(f"Created toggle for channel '{channel_name}' with ID: {new_channel_id}")
                else:
                    logger.error(f"Failed to get channel ID from response: {str(channel_response)[:500]}")
                    return
            except Exception as e:
                logger.error(f"Error creating channel toggle: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return
            
            # Now add content to the channel toggle section
            if new_channel_id:
                # Current date for the update info
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # Create channel content
                channel_content = [
                    # Date info
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": "Updated on "}, "annotations": {"italic": True}},
                                {"type": "text", "text": {"content": current_date}, "annotations": {"bold": True}}
                            ]
                        }
                    },
                    # Divider
                    {
                        "object": "block", 
                        "type": "divider", 
                        "divider": {}
                    }
                ]
                
                # Add summary with enhanced formatting
                if summary:
                    channel_content.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": self._enhance_text_formatting(summary)
                        }
                    })
                
                # Add key points with enhanced formatting
                if isinstance(key_points, list) and key_points:
                    # Add subheading for key points
                    channel_content.append({
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "Key Points"}, "annotations": {"bold": True}}]
                        }
                    })
                    
                    # Add each key point as a bullet with enhanced formatting
                    for point in key_points:
                        channel_content.append({
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": self._enhance_text_formatting(point)
                            }
                        })
                
                # Add bottom update date
                channel_content.append({
                    "object": "block", 
                    "type": "divider", 
                    "divider": {}
                })
                
                channel_content.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": "Last updated: "}, "annotations": {"italic": True}},
                            {"type": "text", "text": {"content": current_date}, "annotations": {"bold": True}}
                        ]
                    }
                })
                
                # Add the content to the channel toggle
                logger.info(f"Adding content to channel '{channel_name}'")
                try:
                    await self.notion.blocks.children.append(
                        block_id=new_channel_id,
                        children=channel_content
                    )
                    logger.info(f"Successfully added content to channel '{channel_name}'")
                except Exception as e:
                    logger.error(f"Error adding content to channel: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error("No Technical Analysis block ID found, unable to update content")
