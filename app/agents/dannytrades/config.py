"""
Configuration management for DannyTrades agent.

This module handles loading and managing configuration settings
for the DannyTrades agent, including API keys, model settings, and MCP server configuration.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
from app.core.config import load_config


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: str = Field(..., description="Provider name (openai, claude, gemini, etc.)")
    model_id: str = Field(..., description="Model identifier")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    endpoint: Optional[str] = Field(None, description="Custom endpoint URL")
    temperature: float = Field(default=0.2, description="Temperature for generation")
    max_tokens: int = Field(default=4000, description="Maximum tokens to generate")
    enabled: bool = Field(default=True, description="Whether this provider is enabled")


class AgentPromptConfig(BaseModel):
    """Configuration for agent prompts."""
    system_prompt: str = Field(..., description="System prompt for the agent")
    processing_instructions: str = Field(..., description="Processing instructions")
    error_handling_prompt: str = Field(..., description="Error handling instructions")


class MCPTransportType(str, Enum):
    """MCP server transport types."""
    STDIO = "stdio"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str = Field(..., description="Server name/identifier")
    transport_type: MCPTransportType = Field(..., description="Transport type (stdio or sse)")
    command: Optional[str] = Field(None, description="Command to run for stdio transport")
    args: Optional[List[str]] = Field(default_factory=list, description="Arguments for stdio command")
    url: Optional[str] = Field(None, description="URL for SSE transport")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")
    enabled: bool = Field(default=True, description="Whether this server is enabled")


class DannyTradesConfig(BaseModel):
    """Configuration class for DannyTrades agent."""
    
    def __init__(self, **data):
        # Load ALL configuration from JSON file only - ignore YAML config
        from app.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        saved_config = config_manager.load_dannytrades_config()
        
        print(f"DEBUG CONFIG: Loading ALL config from JSON: {saved_config}")
        
        # Get current provider from JSON
        current_provider = saved_config.get('current_provider', 'openai')
        providers = saved_config.get('providers', {})
        
        # Use JSON config values only
        data.setdefault('default_provider', current_provider)
        data.setdefault('default_model', providers.get(current_provider, {}).get('model', 'gpt-4'))
        data.setdefault('temperature', 0.2)  # Default values
        data.setdefault('max_tokens', 4000)
        
        # Load API keys from JSON only
        data.setdefault('openai_api_key', providers.get('openai', {}).get('api_key'))
        data.setdefault('anthropic_api_key', providers.get('anthropic', {}).get('api_key'))
        data.setdefault('gemini_api_key', providers.get('gemini', {}).get('api_key'))
        data.setdefault('groq_api_key', providers.get('groq', {}).get('api_key'))
        data.setdefault('mistral_api_key', providers.get('mistral', {}).get('api_key'))
        data.setdefault('zai_api_key', providers.get('zai', {}).get('api_key'))
        
        print(f"DEBUG CONFIG: Using provider={current_provider}")
        print(f"DEBUG CONFIG: Using model={data.get('default_model')}")
        print(f"DEBUG CONFIG: API key loaded={bool(data.get(f'{current_provider}_api_key'))}")
        
        # Call parent constructor
        super().__init__(**data)
        
        # Store JSON config for MCP and other settings
        self._json_config = saved_config
        self._providers_config = {}  # Empty - we use JSON only
        self._dannytrades_config = {}  # Empty - we use JSON only
    
    # LLM Provider Settings
    openai_api_key: Optional[str] = Field(None)
    anthropic_api_key: Optional[str] = Field(None)
    gemini_api_key: Optional[str] = Field(None)
    groq_api_key: Optional[str] = Field(None)
    mistral_api_key: Optional[str] = Field(None)
    zai_api_key: Optional[str] = Field(None)
    
    # LLM Provider configuration from config.yaml
    default_provider: str = Field(description="LLM provider to use")
    default_model: str = Field(description="Model to use")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=4000)
    
    # MCP Server Configuration - loaded from JSON file
    def get_mcp_servers(self) -> List[MCPServerConfig]:
        """Get MCP server configurations from JSON file"""
        mcp_servers_config = self._json_config.get('mcp_servers', {})
        servers = []
        
        print(f"DEBUG MCP: Loading MCP servers from JSON: {mcp_servers_config}")
        
        for server_name, server_config in mcp_servers_config.items():
            if server_config.get('enabled', True):
                # Resolve environment variables in env config
                env = {}
                for key, value in server_config.get('env', {}).items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]  # Remove ${ and }
                        env[key] = os.getenv(env_var, "")
                    else:
                        env[key] = value
                
                print(f"DEBUG MCP: Configured server '{server_name}' with env: {env}")
                
                servers.append(MCPServerConfig(
                    name=server_name,
                    transport_type=server_config.get('transport_type', 'stdio'),
                    command=server_config.get('command'),
                    args=server_config.get('args', []),
                    url=server_config.get('url'),
                    env=env,
                    enabled=server_config.get('enabled', True)
                ))
        
        return servers
    
    @property
    def mcp_servers(self) -> List[MCPServerConfig]:
        """Get MCP servers as property for backward compatibility"""
        return self.get_mcp_servers()
    
    # Legacy Notion Settings (for backward compatibility)
    notion_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("NOTION_API_KEY"))
    notion_database_id: Optional[str] = Field(default_factory=lambda: os.getenv("NOTION_DATABASE_ID"))
    
    # Agent Settings
    max_retries: int = Field(default=3, description="Maximum number of retries for failed operations")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Performance Settings
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    
    def get_mcp_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get MCP server configuration by name."""
        for server in self.mcp_servers:
            if server.name == name and server.enabled:
                return server
        return None
    
    def add_mcp_server(self, server: MCPServerConfig) -> None:
        """Add or update an MCP server configuration."""
        # Remove existing server with same name
        self.mcp_servers = [s for s in self.mcp_servers if s.name != server.name]
        self.mcp_servers.append(server)
    
    def get_notion_database_id(self) -> Optional[str]:
        """Get Notion database ID from JSON MCP server config."""
        # Get from JSON MCP server environment only
        notion_server = self.get_mcp_server("notion")
        if notion_server and notion_server.env:
            db_id = notion_server.env.get("NOTION_DATABASE_ID")
            print(f"DEBUG NOTION: Database ID from JSON MCP config: {db_id}")
            if db_id:
                return db_id
        
        # No fallback - force JSON config usage
        print("DEBUG NOTION: No database ID found in JSON MCP config")
        return None
    
    def switch_provider(self, provider: str) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Provider name to switch to
            
        Returns:
            True if switch was successful
        """
        # For now, just return True as the actual switching logic
        # is handled by the ConfigManager
        return True
    
    def get_current_provider(self) -> str:
        """Get the current LLM provider."""
        return self.default_provider
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider - load from JSON config."""
        from app.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        saved_config = config_manager.load_dannytrades_config()
        json_providers = saved_config.get('providers', {})

        provider_config = json_providers.get(provider, {})

        # Ensure essential keys have default fallbacks if not in the specific provider config
        provider_config.setdefault('model_id', self.default_model)
        provider_config.setdefault('api_key', getattr(self, f'{provider}_api_key', None))
        provider_config.setdefault('enabled', True)

        return provider_config
    
    def validate_llm_config(self) -> tuple[bool, str]:
        """
        Validate that LLM provider is properly configured.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.default_provider:
            return False, "No LLM provider configured. Please set up a provider first."
        
        if not self.default_model:
            return False, f"No model configured for provider '{self.default_provider}'. Please configure a model."
        
        # Check provider-specific requirements (relaxed for Gemini and Ollama to fix server)
        if self.default_provider == "openai":
            # Remove API key validation - allow any API key format for OpenAI/OpenRouter
            pass
        elif self.default_provider == "anthropic" and not self.anthropic_api_key:
            return False, "Anthropic API key is required but not configured."
        elif self.default_provider == "groq" and not self.groq_api_key:
            return False, "Groq API key is required but not configured."
        elif self.default_provider == "gemini":
            # Relaxed validation for Gemini - allow without API key for now
            pass
        elif self.default_provider == "ollama":
            # Ollama doesn't require API key - it's local
            pass
        elif self.default_provider == "mistral" and not self.mistral_api_key:
            return False, "Mistral API key is required but not configured."
        # Note: Gemini and Ollama validation temporarily relaxed
        
        return True, ""
    
    def get_prompts(self) -> Dict[str, str]:
        """Get agent prompts."""
        prompt_manager = PromptManager()
        prompts = prompt_manager.get_prompts()
        return {
            'system_prompt': prompts.system_prompt,
            'processing_instructions': prompts.processing_instructions,
            'error_handling_prompt': prompts.error_handling_prompt
        }
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


class LLMProviderManager:
    """
    Manager class for LLM provider configurations.
    
    This class handles the configuration and management of multiple LLM providers.
    """
    
    def __init__(self):
        self.providers: Dict[str, LLMProviderConfig] = {}
        self._load_default_providers()
    
    def _load_default_providers(self):
        """Load default LLM provider configurations."""
        
        # Ollama configuration
        self.providers["ollama"] = LLMProviderConfig(
            provider="ollama",
            model_id="llama2",
            endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
            temperature=0.2,
            max_tokens=4000,
            enabled=bool(os.getenv("OLLAMA_ENABLED", "false").lower() == "true")
        )
        
        # Claude configuration
        self.providers["claude"] = LLMProviderConfig(
            provider="claude",
            model_id="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            endpoint=os.getenv("CLAUDE_ENDPOINT"),
            temperature=0.2,
            max_tokens=4000,
            enabled=bool(os.getenv("ANTHROPIC_API_KEY"))
        )
        
        # Gemini configuration
        self.providers["gemini"] = LLMProviderConfig(
            provider="gemini",
            model_id="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            max_tokens=4000,
            enabled=bool(os.getenv("GOOGLE_API_KEY"))
        )
        
        # Groq configuration
        self.providers["groq"] = LLMProviderConfig(
            provider="groq",
            model_id="mixtral-8x7b-32768",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
            max_tokens=4000,
            enabled=bool(os.getenv("GROQ_API_KEY"))
        )
        
        # Mistral configuration
        self.providers["mistral"] = LLMProviderConfig(
            provider="mistral",
            model_id="mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.2,
            max_tokens=4000,
            enabled=bool(os.getenv("MISTRAL_API_KEY"))
        )
        
        # OpenAI-compatible configuration
        self.providers["openai"] = LLMProviderConfig(
            provider="openai",
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            endpoint=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
            temperature=0.2,
            max_tokens=4000,
            enabled=bool(os.getenv("OPENAI_API_KEY"))
        )
    
    def get_provider(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get a specific LLM provider configuration."""
        return self.providers.get(provider_name)
    
    def get_enabled_providers(self) -> List[LLMProviderConfig]:
        """Get all enabled LLM providers."""
        return [provider for provider in self.providers.values() if provider.enabled]
    
    def update_provider(self, provider_name: str, config: LLMProviderConfig):
        """Update a provider configuration."""
        self.providers[provider_name] = config
    
    def get_ollama_models(self, endpoint: str) -> List[str]:
        """
        Fetch available models from Ollama endpoint.
        
        Args:
            endpoint: Ollama API endpoint
            
        Returns:
            List of available model names
        """
        try:
            import requests
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
        return []


class MCPServerManager:
    """
    Manager class for MCP server configurations.
    
    This class handles the configuration and management of MCP servers.
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_default_servers()
    
    def _load_default_servers(self):
        """Load default MCP server configurations."""
        
        # Notion MCP server
        self.servers["notion"] = MCPServerConfig(
            name="notion",
            transport_type=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@makenotion/notion-mcp-server"],
            env={
                "NOTION_API_KEY": os.getenv("NOTION_API_KEY", "")
            },
            enabled=bool(os.getenv("NOTION_API_KEY"))
        )
    
    def get_server(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get a specific MCP server configuration."""
        return self.servers.get(server_name)
    
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get all enabled MCP servers."""
        return [server for server in self.servers.values() if server.enabled]
    
    def update_server(self, server_name: str, config: MCPServerConfig):
        """Update a server configuration."""
        self.servers[server_name] = config


class PromptManager:
    """
    Manager class for agent prompts.
    
    This class handles loading and managing customizable prompts for the agent.
    """
    
    def __init__(self):
        self.prompts = self._load_default_prompts()
    
    def _load_default_prompts(self) -> AgentPromptConfig:
        """Load default agent prompts."""
        
        system_prompt = """You are an **expert Notion Stock Database Agent**. Your role is to maintain the Notion database (ID: `{database_id}`) in a way that ensures high precision, accuracy, and contextual completeness of stock entries.

You will receive a structured JSON stock object. Your task is to **either create a new stock entry or update an existing one**, using smart logic that preserves relevant existing data, merges when partial data is received, and fully replaces when input data is complete and more relevant.
Use available tools to update the database."""

        processing_instructions = """
### 📍 Step 1: Identify the Stock

Use `API-post-database-query` to locate the existing stock entry by comparing the input `"symbol"` with the `"Stock Ticker"` property in the database.

```json
{
  "database_id": "{database_id}",
  "filter": {
    "property": "Stock Ticker",
    "title": {
      "equals": "symbol"
    }
  }
}
```

### 🔄 Step 2: Update Existing Stock Entry

If a page is found:

1. Extract its `page_id`.
2. Use the `API-patch-page` to update only the relevant fields.
3. Apply intelligent update rules:

- **If input data is complete and clearly more relevant**, **replace** existing values.
- **If input is partial**, **merge** with existing values. Do not drop prior context unless it is explicitly outdated or irrelevant.
- When merging insights (e.g. red candles, summaries), **combine both new and existing data** to preserve full understanding.
- Output should always be **more informative than before**.

#### Example Tool Payload:
```json
{
  "page_id": "PAGE_ID_OF_EXISTING_STOCK",
  "properties": {
    "Red/Yellow Candle": {
      "multi_select": [
        { "name": "Red-Daily" },
        { "name": "Red-Weekly" },
        { "name": "Red-Monthly" },
        { "name": "Yellow-Daily" },
        { "name": "Yellow-Weekly" }
      ]
    },
    "Volatility Hole": {
      "checkbox": true
    },
    "Whale Accumulation": {
      "rich_text": [
        { "text": { "content": "Combined whale accumulation insight..." } }
      ]
    },
    "Important Details": {
      "rich_text": [
        { "text": { "content": "Merged summary of existing + new info" } }
      ]
    },
    "Hot Stocks": {
      "multi_select": [
        { "name": "Hot" },
        { "name": "Extreme Bullish" },
        { "name": "Bearish" }
      ]
    },
    "Link": {
      "url": "https://example.com/patreon"
    },
    "Support/Resistance": {
      "rich_text": [
        { "text": { "content": "Support Levels: 101, Resistance Levels: 120" } }
      ]
    }
  }
}
```

### 🆕 Step 3: Create New Stock Entry

If no matching page is found, use the `API-post-page` tool to insert a new stock record.

#### Tool Payload:
```json
{
  "parent": { "database_id": "{database_id}" },
  "properties": {
    "Stock Ticker": {
      "title": [
        { "text": { "content": "SYMBOL" } }
      ]
    },
    "Red/Yellow Candle": {
      "multi_select": [
        { "name": "Red-Daily" },
        { "name": "Red-Weekly" },
        { "name": "Red-Monthly" },
        { "name": "Yellow-Daily" },
        { "name": "Yellow-Weekly" }
      ]
    },
    "Volatility Hole": {
      "checkbox": true
    },
    "Whale Accumulation": {
      "rich_text": [
        { "text": { "content": "Whale accumulation text" } }
      ]
    },
    "Important Details": {
      "rich_text": [
        { "text": { "content": "Stock summary and key observations" } }
      ]
    },
    "Hot Stocks": {
      "multi_select": [
        { "name": "Hot" },
        { "name": "Extreme Bullish" },
        { "name": "Bearish" }
      ]
    },
    "Support/Resistance": {
      "rich_text": [
        { "text": { "content": "Support Levels: X, Resistance Levels: Y" } }
      ]
    },
    "Link": {
      "url": "https://example.com/patreon"
    }
  }
}
```

### 🧠 Field Mapping Logic

| Notion Property         | Input Key(s)                          | Type         | Smart Update Strategy |
|-------------------------|---------------------------------------|--------------|------------------------|
| `Red/Yellow Candle`     | `red_candle`, `yellow_candle`         | Multi-select | Merge values unless new data fully replaces. |
| `Volatility Hole`       | `volatility_hole`                     | Checkbox     | Set to `true` if present, else `false`. |
| `Whale Accumulation`    | `whale_accumulation`                  | Rich Text    | Merge or replace based on insight completeness. |
| `Hot Stocks`            | `hot_stocks`                          | Multi-select | Merge and deduplicate. |
| `Important Details`     | `misc_data`                           | Rich Text    | Combine summaries. Prefer unified explanation. |
| `Support/Resistance`    | `support_levels`, `resistance_levels` | Rich Text    | Format: `"Support Levels: ..., Resistance Levels: ..."` |
| `Link`                  | `patreon_link`                        | URL          | Replace if newer or more complete. |

### ⚠️ Special Considerations

- Do not blindly replace valuable data when input is **partial or vague**.
- Be precise when merging — avoid duplicate or conflicting insights.
- If existing data contains summaries, and new input does not mention those, **retain existing summaries**.
- Always aim to produce a **comprehensive and updated snapshot** of the stock.

### 🔚 Final Step

Once updated or created, confirm the data integrity by checking for:

- Missing critical fields
- Improper formatting (e.g., empty select/multi-selects)
- Overwritten context without justification"""

        error_handling_prompt = """
### ⚠️ Error Handling

- If API calls fail, retry up to 3 times with exponential backoff
- Log all errors with detailed context for debugging
- If a stock cannot be processed, continue with the next one
- Always provide a summary of successful and failed operations
- Never lose data due to processing errors"""

        return AgentPromptConfig(
            system_prompt=system_prompt,
            processing_instructions=processing_instructions,
            error_handling_prompt=error_handling_prompt
        )
    
    def get_prompts(self) -> AgentPromptConfig:
        """Get current agent prompts."""
        return self.prompts
    
    def update_prompts(self, prompts: AgentPromptConfig):
        """Update agent prompts."""
        self.prompts = prompts
    
    def get_formatted_system_prompt(self, database_id: str) -> str:
        """Get formatted system prompt with database ID."""
        return self.prompts.system_prompt.format(database_id=database_id)
