"""
Unified DannyTrades Agent using Agno and a single MCP-based approach.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from textwrap import dedent
from typing import Dict, Any, Optional, List

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.anthropic import Claude
from agno.models.ollama import Ollama
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters
import logging

from .models import ProcessingResult, AgentActivity
from .config import DannyTradesConfig, MCPServerConfig

logger = logging.getLogger(__name__)


class SimpleDannyTradesAgent:
    """
    A unified DannyTrades agent that uses the Agno framework and a single,
    correctly implemented MCP tool-calling pattern for all LLM providers.
    """

    def __init__(self, config: Optional[DannyTradesConfig] = None):
        """Initialize the agent, reloading the config each time for freshness."""
        self.config = DannyTradesConfig()
        self.current_provider = self.config.default_provider
        self.current_model = self._create_model_from_config()
        self.activity_log: List[AgentActivity] = []
        self.initialized = True

    def _create_model_from_config(self):
        """Creates an LLM model instance based on the current configuration."""
        provider = self.config.default_provider
        provider_config = self.config.get_provider_config(provider)
        logger.info(f"Loading provider '{provider}' with config: {provider_config}")

        if not provider_config or not provider_config.get('enabled', False):
            raise ValueError(f"Provider {provider} is not available or enabled.")

        model_id = provider_config.get('model_id')
        api_key = provider_config.get('api_key')
        endpoint = provider_config.get('endpoint')
        temperature = self.config.temperature
        max_tokens = self.config.max_tokens

        if provider == "claude":
            return Claude(id=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
        elif provider == "gemini":
            return Gemini(id=model_id, api_key=api_key, temperature=temperature)
        elif provider == "ollama":
            return Ollama(id=model_id, host=endpoint or "http://localhost:11434")

        # Default to OpenAI-compatible, which covers 'openai', 'zai', and others
        headers = {}
        if provider == "zai":
            logger.info("Building special headers for Z.ai/opencode provider.")
            headers = {
                "User-Agent": provider_config.get('user_agent', 'opencode/1.0'),
                "Accept-Language": provider_config.get('accept_language', 'en-US,en'),
                "X-Client-Type": provider_config.get('client_type', 'coding-agent'),
                "X-Agent-Name": provider_config.get('agent_name', 'opencode'),
                "X-Agent-Version": provider_config.get('agent_version', '1.0'),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

        import httpx
        # Always create a client, even if headers are empty, for consistency.
        http_client = httpx.Client(headers=headers)

        logger.info(f"Creating OpenAI-compatible model for '{provider}' at endpoint '{endpoint}'")
        return OpenAIChat(
            id=model_id,
            api_key=api_key,
            base_url=endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            http_client=http_client
        )

    async def process_stock_data(self, stock_data: Dict[str, Any]) -> ProcessingResult:
        """
        Processes stock data using a unified MCP-based approach for all providers.
        """
        start_time = time.time()
        logger.info(f"Processing stock data for '{stock_data.get('symbol', 'Unknown')}' with provider: {self.current_provider}")

        try:
            # 1. Get Notion MCP server configuration from the unified config
            notion_mcp_config = self.config.get_mcp_server("notion")
            if not notion_mcp_config or not notion_mcp_config.enabled:
                raise ValueError("Notion MCP server is not configured or enabled.")
            
            logger.info(f"Found Notion MCP server config: {notion_mcp_config.name}")

            # 2. Set up MCP server parameters, starting with the parent environment
            mcp_env = os.environ.copy()
            # Update with variables from config
            mcp_env.update(notion_mcp_config.env)

            notion_api_key = mcp_env.get("NOTION_API_KEY")
            if notion_api_key:
                # The server may expect the key via OPENAPI_MCP_HEADERS
                mcp_env["OPENAPI_MCP_HEADERS"] = json.dumps({
                    "Authorization": f"Bearer {notion_api_key}",
                    "Notion-Version": "2022-06-28"
                })
                # Remove the old key to avoid conflicts
                if "NOTION_API_KEY" in mcp_env:
                    del mcp_env["NOTION_API_KEY"]

            server_params = StdioServerParameters(
                command=notion_mcp_config.command,
                args=notion_mcp_config.args,
                env=mcp_env
            )

            # 3. Use async context manager for the MCPTools lifecycle
            async with MCPTools(server_params=server_params) as mcp_tools:
                # 4. Get formatted prompts and adjust for Z.ai if necessary
                instructions = self._get_agent_instructions()
                user_prompt = self._get_user_prompt(stock_data)

                if self.current_provider == "zai":
                    # For Z.ai, combine instructions into the user prompt to avoid 'system' role issues
                    logger.info("Z.ai provider detected. Combining instructions into user prompt.")
                    final_prompt = instructions + "\n\n---\n\n" + user_prompt
                    agent_instructions = None
                else:
                    final_prompt = user_prompt
                    agent_instructions = instructions

                # 5. Create the Agno agent
                agent = Agent(
                    model=self.current_model,
                    tools=[mcp_tools],
                    description="An expert Notion Stock Database Agent.",
                    instructions=agent_instructions,
                    markdown=True,
                    show_tool_calls=True,
                )

                # 6. Run the agent
                logger.info(f"Running agent for '{stock_data.get('symbol')}'...")
                result = await agent.arun(final_prompt)
                
                reasoning = str(result.data) if hasattr(result, 'data') else str(result)
                logger.info(f"Agent run completed. Result: {reasoning[:200]}...")

            processing_time = time.time() - start_time
            result = ProcessingResult(
                success=True,
                reasoning=reasoning,
                processing_time=processing_time,
                stock_symbol=stock_data.get("symbol", "Unknown"),
                action="analysis_and_update"
            )

        except Exception as e:
            import traceback
            error_message = f"Error processing stock data: {e}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            processing_time = time.time() - start_time
            result = ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                stock_symbol=stock_data.get("symbol", "Unknown")
            )
        
        # Add a record to the activity log
        activity = AgentActivity(
            activity_id=str(uuid.uuid4()),
            activity_type="process_stock_data",
            stock_symbol=stock_data.get("symbol", "Unknown"),
            message=f"Processed with {self.current_provider}. Success: {result.success}",
            timestamp=datetime.now(),
            status="success" if result.success else "error"
        )
        self.activity_log.append(activity)

        return result

    def _get_agent_instructions(self) -> str:
        """Formats the system prompt with the correct database ID."""
        prompts = self.config.get_prompts()
        database_id = self.config.get_notion_database_id()
        if not database_id:
            raise ValueError("Notion Database ID is not configured.")

        # Use .replace() for safer substitution in prompts with JSON examples
        system_prompt = prompts['system_prompt'].replace('{database_id}', database_id)
        processing_instructions = prompts['processing_instructions'].replace('{database_id}', database_id)

        return system_prompt + "\n" + processing_instructions

    def _get_user_prompt(self, stock_data: Dict[str, Any]) -> str:
        """Creates the user-facing prompt with the stock data."""
        return f"""Here is the stock data to process:

```json
{json.dumps(stock_data, indent=2)}
```

Please follow your instructions to process this data and update the Notion database.
"""

    async def switch_provider(self, provider: str, **kwargs) -> bool:
        """Switches the LLM provider and re-initializes the model."""
        try:
            from app.core.config_manager import ConfigManager
            config_manager = ConfigManager()

            logger.info(f"Attempting to save new provider '{provider}' to config...")
            success = config_manager.update_provider_config(provider=provider, **kwargs)

            if not success:
                logger.error("Failed to save provider configuration to JSON file.")
                return False

            logger.info(f"Configuration saved. Reloading agent with new provider '{provider}'...")

            # Reload config and model
            self.config = DannyTradesConfig()
            self.current_provider = self.config.default_provider
            self.current_model = self._create_model_from_config()

            # Verify the switch
            if self.current_provider != provider:
                logger.error(f"Failed to switch provider. Current provider is '{self.current_provider}' after reload, expected '{provider}'.")
                return False

            logger.info(f"Successfully switched to and reloaded with '{provider}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to switch provider due to an exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Gets the current status of the agent."""
        return {
            "status": "active",
            "provider": self.current_provider,
            "model": getattr(self.current_model, 'id', 'unknown'),
            "initialized": self.initialized,
        }

    async def get_activities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Gets recent agent activities."""
        recent_activities = self.activity_log[-limit:] if limit else self.activity_log
        return [
            {
                "activity_id": activity.activity_id,
                "timestamp": activity.timestamp.isoformat(),
                "activity_type": activity.activity_type,
                "stock_symbol": activity.stock_symbol,
                "message": activity.message,
                "status": activity.status
            }
            for activity in recent_activities
        ]

    def get_mcp_server_status(self) -> Dict[str, Any]:
        """Gets the status of the MCP server."""
        notion_mcp_config = self.config.get_mcp_server("notion")
        is_configured = notion_mcp_config and notion_mcp_config.enabled
        return {
            "notion": {
                "status": "available" if is_configured else "not_configured",
                "type": "official_mcp_server"
            }
        }

    def get_provider_status(self) -> Dict[str, Any]:
        """Gets the status of the current LLM provider."""
        # Get all provider names from the config's providers section
        provider_keys = self.config._json_config.get('providers', {}).keys()
        available_providers = list(provider_keys)

        return {
            "current": self.current_provider,
            "model_name": getattr(self.current_model, 'id', 'unknown'),
            "available": available_providers,
            "status": "active"
        }

    async def test_analysis(self, symbol: str = "TEST") -> Dict[str, Any]:
        """Runs a test analysis with sample data."""
        test_data = {
            "symbol": symbol,
            "red_candle": {"daily": True},
            "yellow_candle": {"weekly": True},
            "volatility_hole": True,
            "whale_accumulation": "Detected significant whale activity.",
            "hot_stocks": ["Hot", "Extreme Bullish"],
            "misc_data": "This is a test entry for system verification.",
            "support_levels": [100, 95],
            "resistance_levels": [110, 115],
            "patreon_link": "https://www.patreon.com/dannytrades"
        }
        result = await self.process_stock_data(test_data)
        return {
            "status": "success" if result.success else "error",
            "message": result.reasoning if result.success else result.error_message,
            "symbol": symbol,
            "provider": self.current_provider,
            "processing_time": result.processing_time
        }

# Global instance for web API
simple_agent = SimpleDannyTradesAgent()

async def process_stock_simple(stock_data: Dict[str, Any]) -> ProcessingResult:
    """Simple function for processing stock data, callable from FastAPI."""
    return await simple_agent.process_stock_data(stock_data)