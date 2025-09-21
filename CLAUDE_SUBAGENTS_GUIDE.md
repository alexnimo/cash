# Claude Subagents Setup Guide

This guide explains how to set up and work with Claude subagents, specifically focusing on the correct installation of the Notion MCP server using the stdio version (not the streaming version which has formatting issues).

## Prerequisites

1. **Claude CLI installed** - Install from https://docs.anthropic.com/claude/docs/claude-cli
2. **Notion API token** - Create an integration at https://www.notion.so/my-integrations
3. **Node.js installed** - For running the Notion MCP server

## Installation

### 1. Install Claude CLI

```bash
# Install Claude CLI
curl -fsSL https://claude.ai/install.sh | sh

# Authenticate
claude auth login
```

### 2. Install Notion MCP Server (stdio version)

**IMPORTANT**: Use the stdio version, NOT the streaming version which has formatting issues.

```bash
# Add the Notion MCP server using stdio
claude mcp add-json notionApi "{\"type\":\"stdio\",\"command\":\"npx\",\"args\":[\"-y\",\"@notionhq/notion-mcp-server\"],\"env\":{\"NOTION_TOKEN\":\"YOUR_NOTION_TOKEN_HERE\"}}"
```

Replace `YOUR_NOTION_TOKEN_HERE` with your actual Notion API token.

### 3. Verify Installation

```bash
# Check available MCP servers
claude mcp list

# Should show notionApi in the list
```

## Working with Claude Subagents

### Available Subagents

1. **danny-trades** - Stock analysis and market data processing
2. **notionApi** - Notion database operations via MCP server

### Using the danny-trades Subagent

The danny-trades subagent is designed to process stock market data and generate technical analysis reports.

#### Data Format

```json
{
  "symbol": "AAPL",
  "red_candle": {
    "daily": true,
    "weekly": false
  },
  "yellow_candle": ["daily"],
  "volatility_hole": true,
  "whale_accumulation": "Daily: 61.20%, Weekly: 53.5%",
  "hot_stocks": ["Extreme Bullish"],
  "misc_data": "Summary of observations",
  "support_levels": [222, 225],
  "resistance_levels": [235, 238],
  "patreon_link": "https://www.patreon.com/posts/your-link"
}
```

#### Example Usage

```bash
# Basic stock analysis
claude -p "Use the danny-trades subagent to process this data: {\"symbol\": \"AAPL\", \"red_candle\": {\"daily\": true}, \"misc_data\": \"AAPL shows bullish signal\"}"

# With verbose logging
claude -p "Use the danny-trades subagent to process this data: {\"symbol\": \"TSLA\", \"volatility_hole\": true}" --verbose
```

### Integration with Your Application

The CLI agent in your application automatically handles:

1. **JSON serialization** - Converts Python objects to JSON strings
2. **Prompt formatting** - Creates the correct prompt structure
3. **Command execution** - Runs Claude CLI with proper flags
4. **Output parsing** - Processes Claude's response
5. **Error handling** - Handles timeouts and failures

## Configuration

### CLI Agent Configuration

The CLI agent is configured in `app/agents/dannytrades/cli_config.py`:

```python
"claude": AgentConfig(
    name="claude",
    command="claude",  # Path to claude executable
    args=[],  # Additional arguments
    env_vars={},  # Environment variables
    timeout=60,  # Timeout in seconds
    enabled=True  # Enable/disable the agent
)
```

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Claude CLI configuration
export CLAUDE_API_KEY="your-claude-api-key"

# Notion API token for MCP server
export NOTION_TOKEN="your-notion-integration-token"

# Node.js path (if needed)
export PATH="$PATH:/usr/local/bin"
```

## Troubleshooting

### Common Issues

1. **Claude CLI not found**
   ```bash
   # Check if claude is installed
   which claude

   # Add to PATH if needed
   export PATH="$PATH:/path/to/claude"
   ```

2. **MCP server not working**
   ```bash
   # Check MCP server status
   claude mcp list

   # Remove and re-add if needed
   claude mcp remove notionApi
   claude mcp add-json notionApi "{\"type\":\"stdio\",\"command\":\"npx\",\"args\":[\"-y\",\"@notionhq/notion-mcp-server\"],\"env\":{\"NOTION_TOKEN\":\"YOUR_TOKEN\"}}"
   ```

3. **Permission issues**
   ```bash
   # Fix npm permissions
   npm config set prefix ~/.npm-global
   export PATH=~/.npm-global/bin:$PATH
   ```

4. **Notion token issues**
   - Ensure your Notion integration has access to the required databases
   - Verify the token is correct and not expired
   - Check that the integration is shared with your workspace

### Debug Commands

```bash
# Test Claude CLI directly
claude --help

# Test with simple prompt
claude -p "Hello, world!"

# Check MCP server configuration
claude mcp list

# Test Notion MCP server
claude -p "Use the notionApi subagent to list available databases"
```

## Best Practices

1. **Use stdio MCP server** - Always prefer stdio over streaming for better compatibility
2. **Proper error handling** - Always handle timeouts and failures in your code
3. **Environment management** - Use environment variables for sensitive data
4. **Logging** - Enable verbose logging for debugging
5. **Timeout management** - Set appropriate timeouts for your use case

## Windows-Specific Notes

When running on Windows:

1. **Use full paths** - Specify the complete path to the claude executable
2. **Escape special characters** - JSON data needs proper escaping for command line
3. **Use shell=True** - Required for proper command execution on Windows
4. **Node.js installation** - Ensure Node.js is in your system PATH

Example Windows path configuration:
```python
"claude": AgentConfig(
    name="claude",
    command="C:\\nvm4w\\nodejs\\claude.CMD",
    args=[],
    env_vars={},
    timeout=60,
    enabled=True
)
```

## Integration Example

```python
from app.agents.dannytrades.cli_agent import CLIAgent

# Initialize agent
agent = CLIAgent()

# Prepare stock data
data = {
    "symbol": "AAPL",
    "red_candle": {"daily": True},
    "misc_data": "Bullish signal detected"
}

# Analyze using Claude CLI
result = await agent.analyze(data)

if result["success"]:
    print("Analysis result:", result["result"])
else:
    print("Error:", result["error"])
```

## Security Considerations

1. **Token security** - Never commit API tokens to version control
2. **Environment isolation** - Use different environments for development and production
3. **Access control** - Limit Notion integration permissions to only what's necessary
4. **Input validation** - Always validate and sanitize input data
5. **Error messages** - Avoid exposing sensitive information in error outputs