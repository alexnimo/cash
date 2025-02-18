from llama_index.core import Response
from llama_index.core.tools import BaseTool, ToolMetadata
from notion_client import AsyncClient
from typing import Dict, List, Any, Optional
import os
import logging
import yaml
import uuid
from datetime import datetime
from pathlib import Path
from app.core.settings import get_settings

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class NotionAdvancedToolSpec(BaseTool):
    """Advanced Notion tool for interacting with Notion databases and pages."""
    metadata = ToolMetadata(
        name="notion_advanced_tool",
        description="Advanced Notion tool for interacting with Notion databases and pages"
    )

    def __init__(self):
        # Initialize parent class first
        super().__init__()
        
        # Load config
        config = load_config()
        settings = get_settings()
        
        # Initialize Notion client
        self.notion = AsyncClient(auth=settings.notion.api_key)
        
        # Get database ID from settings
        self.database_id = settings.notion.database_id
        if not self.database_id:
            raise ValueError("Notion database ID is not configured")
            
        # Get property names from config.yaml
        notion_config = config.get('notion', {})
        self.stock_ticker_property = notion_config.get('stock_ticker_property')
        self.charts_property = notion_config.get('charts_property')
        self.ta_summary_property = notion_config.get('ta_summary_property')
        
        # Validate config
        if not all([self.stock_ticker_property, self.charts_property, self.ta_summary_property]):
            raise ValueError("Missing required Notion property configurations in config.yaml")

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
            name="notion_advanced_tool",
            description="Advanced Notion tool for interacting with Notion databases and pages",
            fn_schema={
                "type": "function",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform (create_page, update_page, query_database)",
                    },
                    "data": {
                        "type": "object",
                        "description": "Data for the operation",
                    }
                },
                "required": ["operation", "data"]
            }
        )

    async def __call__(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with given operation and data"""
        try:
            if operation == "create_page":
                return await self.create_page(data)
            elif operation == "update_page":
                return await self.update_page(data)
            elif operation == "query_database":
                return await self.query_database(data)
            elif operation == "create_technical_analysis_section":
                return await self.create_technical_analysis_section(data)
            elif operation == "upload_chart_image":
                return await self.upload_chart_image(data)
            elif operation == "update_market_insights":
                return await self.update_market_insights(data)
            elif operation == "create_or_update_stock_page":
                return await self.create_or_update_stock_page(data)
            elif operation == "get_all_tickers":
                return await self.get_all_tickers()
            elif operation == "update_technical_analysis":
                return await self.update_technical_analysis(data["page_id"], data["content"], data["channel_name"])
            elif operation == "add_chart_to_page":
                return await self.add_chart_to_page(data["page_id"], data["image_path"])
            elif operation == "get_stock_page":
                return await self.get_stock_page(data["ticker"])
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in NotionAdvancedToolSpec: {str(e)}")
            raise

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
            page_id = data.pop("page_id")
            response = await self.notion.pages.update(
                page_id=page_id,
                properties=data
            )
            return {"status": "success", "page_id": response["id"]}
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
                        "property": self.stock_ticker_property,
                        "direction": "ascending"
                    }
                ]
            })
            
            tracked_stocks = []
            for page in response.get('results', []):
                symbol = page.get('properties', {}).get(self.stock_ticker_property, {}).get('rich_text', [])
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
        """Create or update a stock page in the database"""
        try:
            ticker = data["ticker"]
            content = data["content"]
            # Search for existing page
            results = await self.notion.databases.query(
                database_id=self.database_id,
                filter={
                    "property": "Stock Ticker",
                    "title": {"equals": ticker}
                }
            )
            
            if results["results"]:
                page_id = results["results"][0]["id"]
                await self.update_stock_page(page_id, content)
                return {"status": "success", "page_id": page_id}
            else:
                return await self.create_stock_page(ticker, content)
        except Exception as e:
            logger.error(f"Error creating/updating stock page: {str(e)}")
            raise

    async def create_stock_page(self, ticker: str, content: Dict) -> Dict:
        """Create a new stock page"""
        try:
            page = await self.notion.pages.create(
                parent={"database_id": self.database_id},
                properties={
                    "Stock Ticker": {
                        "title": [{"text": {"content": ticker}}]
                    },
                    "Last Updated": {
                        "date": {
                            "start": datetime.now().isoformat()
                        }
                    }
                }
            )
            
            if content.get("technical_analysis"):
                await self.create_technical_analysis_section({"page_id": page["id"], "content": content["technical_analysis"]})
            
            if content.get("market_insights"):
                await self.update_market_insights({"page_id": page["id"], "insights": content["market_insights"]})
                
            if content.get("chart_url"):
                await self.upload_chart_image({"page_id": page["id"], "image_url": content["chart_url"]})
                
            return {"status": "success", "page_id": page["id"]}
        except Exception as e:
            logger.error(f"Error creating stock page: {str(e)}")
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
                            break
                            
            logger.info(f"Found {len(tracked_stocks)} tracked stocks")
            return tracked_stocks
            
        except Exception as e:
            logger.error(f"Error getting tickers from Notion: {str(e)}")
            raise
            
    async def update_technical_analysis(self, page_id: str, content: str, channel_name: str) -> bool:
        """Update technical analysis content for a stock page."""
        try:
            blocks = [
                {
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": f"Technical Analysis Update ({channel_name})"}}]
                    }
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": content}}]
                    }
                }
            ]
            
            await self.notion.blocks.children.append(page_id, blocks)
            return True
        except Exception as e:
            logger.error(f"Error updating technical analysis: {str(e)}")
            return False
            
    async def add_chart_to_page(self, page_id: str, image_path: str) -> bool:
        """Add a chart image to a Notion page."""
        try:
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return False
                
            # Upload image to S3 or another storage service first
            # Then create an image block with the URL
            image_url = await self._upload_image(image_path)
            
            blocks = [{
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {
                        "url": image_url
                    }
                }
            }]
            
            await self.notion.blocks.children.append(page_id, blocks)
            return True
        except Exception as e:
            logger.error(f"Error adding chart to page: {str(e)}")
            return False
            
    async def get_stock_page(self, ticker: str) -> Optional[Dict]:
        """Get the Notion page for a specific stock ticker."""
        try:
            query_result = await self.query_database(filter={
                "property": "Ticker",
                "title": {
                    "equals": ticker
                }
            })
            
            if not query_result.get('results'):
                return None
                
            return query_result['results'][0]
        except Exception as e:
            logger.error(f"Error getting stock page: {str(e)}")
            return None
            
    async def _upload_image(self, image_path: str) -> str:
        """Upload an image to storage and return the URL."""
        # Implement image upload to S3 or another storage service
        # For now, return a placeholder URL
        return f"https://storage.example.com/{Path(image_path).name}"
