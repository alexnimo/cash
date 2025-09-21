# DannyTrades Agent Refactoring Task

## 🎯 Objective
Refactor the SimpleDannyTradesAgent to properly separate Z.AI and Agno providers while ensuring unified MCP server usage and consistent specialized prompts.

## 🔍 Current Issues

### 1. Z.AI Provider Scattered Across Multiple Functions
- `_process_with_zai_direct()` - Does custom HTTP request to Z.AI
- `_process_with_mcp_server()` - Handles MCP integration separately
- **Problem**: Z.AI logic should be unified in a single function

### 2. Agno Providers Broken
- Line 189: `tools=[mcp_tools]` - `mcp_tools` is undefined
- Missing proper MCP server initialization
- Using hardcoded simple prompts instead of specialized config prompts

### 3. Inconsistent Prompt Usage
- Agno path: Simple hardcoded prompt (lines 197-206)
- Z.AI path: Different MCP prompt (lines 409-419)
- Config has specialized prompts but not used consistently

### 4. MCP Server Duplication
- Different MCP setup approaches for different providers
- Should use unified MCP server configuration

## 🏗️ Target Architecture

```
SimpleDannyTradesAgent
├── process_stock_data() - Main routing
│   ├── if provider == "zai": _process_with_zai_unified()
│   └── else: _process_with_agno_providers()
├── _process_with_zai_unified() - Single Z.AI function
│   ├── 1. Custom HTTP request to Z.AI API
│   ├── 2. Get analysis response
│   ├── 3. Initialize unified MCP server
│   ├── 4. Update Notion database
│   └── 5. Return ProcessingResult
├── _process_with_agno_providers() - Agno providers
│   ├── 1. Use self.current_model (Agno Agent)
│   ├── 2. Initialize unified MCP server
│   ├── 3. Use specialized prompts from config
│   └── 4. Return ProcessingResult
└── _setup_unified_mcp_server() - Shared MCP setup
```

## 📋 Implementation Steps

### Phase 1: Create Unified MCP Server Setup
- [ ] Create `_setup_unified_mcp_server()` method
- [ ] Use same `@notionhq/notion-mcp-server` for all providers
- [ ] Handle Notion API key validation
- [ ] Return configured MCP tools for agent usage

### Phase 2: Implement Unified Z.AI Function
- [ ] Create `_process_with_zai_unified()` method
- [ ] Preserve exact custom HTTP request logic (headers, auth, endpoint)
- [ ] Integrate MCP server setup within same function
- [ ] Use specialized prompts from config (`self._get_agent_instructions()`)
- [ ] Handle both Z.AI analysis AND Notion database update
- [ ] Return unified `ProcessingResult`

### Phase 3: Fix Agno Providers Function
- [ ] Create `_process_with_agno_providers()` method
- [ ] Use existing `self.current_model` (OpenAI, Claude, Gemini, etc.)
- [ ] Integrate unified MCP server setup
- [ ] Use specialized prompts from config
- [ ] Create Agno Agent with proper MCP tools
- [ ] Return unified `ProcessingResult`

### Phase 4: Update Main Routing
- [ ] Simplify `process_stock_data()` to route between two provider functions
- [ ] Remove `_process_with_mcp_server()` method entirely
- [ ] Ensure consistent error handling across both paths

### Phase 5: Testing & Validation
- [ ] Test Z.AI provider end-to-end functionality
- [ ] Test each Agno provider (OpenAI, Claude, Gemini, Ollama)
- [ ] Validate MCP server integration works for all providers
- [ ] Verify specialized prompts are used consistently
- [ ] Test error handling and edge cases

## 🔧 Key Preservations

### ✅ Must Preserve
- **Z.AI Custom HTTP Request**: Exact same logic, headers, authentication
- **Provider Separation**: Clear distinction between Z.AI and Agno providers
- **Specialized Config Prompts**: Use `PromptManager` from config
- **All Existing Functionality**: No breaking changes to API or behavior
- **ProcessingResult Format**: Consistent return type

### ✅ Must Unify
- **MCP Server Setup**: Same Notion integration for all providers
- **Prompt Usage**: Use config-based specialized prompts consistently
- **Error Handling**: Consistent error response format
- **Logging**: Unified logging approach

## 📁 Files to Modify

### Primary Implementation
- `app/agents/dannytrades/simple_agent.py` - Main refactoring

### Configuration (if needed)
- `app/agents/dannytrades/config.py` - Ensure prompts are accessible
- `app/core/config_manager.py` - Validate config loading

### Testing
- Create test scripts to validate all provider functionality
- Test MCP server integration
- Validate prompt consistency

## 🚀 Success Criteria

1. **Z.AI Provider**: Single function handles custom HTTP + MCP integration
2. **Agno Providers**: Proper MCP integration with specialized prompts
3. **Unified MCP**: Same Notion server setup across all providers
4. **Consistent Prompts**: Config-based specialized prompts used everywhere
5. **No Regressions**: All existing functionality preserved
6. **Clean Architecture**: Clear separation of concerns, no scattered logic

## 📝 Implementation Notes

- Use UV for running tests: `uv run python test_script.py`
- Follow existing error handling patterns
- Maintain existing logging format and levels
- Preserve all existing configuration options
- Ensure backward compatibility with existing API calls
