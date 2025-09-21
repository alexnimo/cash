"""
Custom Ollama client with tool calling support.

This bypasses Agno's Ollama integration to enable proper tool calling
while maintaining compatibility with the Agno Agent framework.
"""

import json
import logging
import httpx
from typing import Dict, Any, List, Optional, AsyncGenerator, Iterator
from agno.models.base import Model

logger = logging.getLogger(__name__)


class CustomOllamaModel(Model):
    """
    Custom Ollama model that supports tool calling via direct API integration.
    
    This class inherits from Agno's base Model class to maintain compatibility
    with the Agent framework while implementing direct Ollama API calls.
    """
    
    def __init__(self, id: str, host: str = "http://localhost:11434", **kwargs):
        """
        Initialize custom Ollama model.
        
        Args:
            id: Model identifier (e.g., "qwen3-14b")
            host: Ollama server endpoint
        """
        super().__init__(id=id, **kwargs)
        self.host = host.rstrip('/')
        self.client = httpx.AsyncClient(timeout=300.0)
        
    async def ainvoke(self, *args, **kwargs) -> str:
        """Async invoke method required by Agno base class."""
        # Handle both positional and keyword arguments
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get('prompt', '')
        
        tools = kwargs.get('tools')
        return await self.arun(prompt, tools)
    
    def invoke(self, *args, **kwargs) -> str:
        """Sync invoke method required by Agno base class."""
        import asyncio
        return asyncio.run(self.ainvoke(*args, **kwargs))
    
    async def ainvoke_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Async streaming invoke method required by Agno base class."""
        # For simplicity, yield the full response at once
        response = await self.ainvoke(prompt, **kwargs)
        yield response
    
    def invoke_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Sync streaming invoke method required by Agno base class."""
        import asyncio
        
        async def _async_gen():
            async for chunk in self.ainvoke_stream(prompt, **kwargs):
                yield chunk
        
        # Convert async generator to sync iterator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gen = _async_gen()
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
    
    def parse_provider_response(self, response: Any, **kwargs) -> str:
        """Parse provider response required by Agno base class."""
        # Ignore response_format and other kwargs that may be passed by Agno
        if isinstance(response, dict):
            if "message" in response:
                return response["message"].get("content", "")
            return response.get("response", str(response))
        return str(response)
    
    def parse_provider_response_delta(self, response: Any, **kwargs) -> str:
        """Parse provider response delta required by Agno base class."""
        return self.parse_provider_response(response, **kwargs)

    async def arun(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Execute model with tool calling support via direct Ollama API.
        
        Args:
            prompt: The prompt to send to the model
            tools: List of available tools in Ollama format
            
        Returns:
            Model response as string
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            
            # Add tools if provided
            if tools:
                ollama_tools = self._convert_agno_tools_to_ollama(tools)
                payload["tools"] = ollama_tools
                logger.info(f"Added {len(ollama_tools)} tools to Ollama request")
            
            # Make the API call
            response = await self.client.post(
                f"{self.host}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle tool calls if present
            if "message" in result:
                message = result["message"]
                
                # Check if model wants to call tools
                if "tool_calls" in message and message["tool_calls"]:
                    return await self._handle_tool_calls(message, tools, prompt)
                
                # Return regular response
                return message.get("content", "")
            
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def _convert_agno_tools_to_ollama(self, agno_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Agno/MCP tools to Ollama tool format.
        
        Args:
            agno_tools: Tools in Agno/MCP format
            
        Returns:
            Tools in Ollama format
        """
        ollama_tools = []
        
        for tool in agno_tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            ollama_tools.append(ollama_tool)
        
        return ollama_tools
    
    async def _handle_tool_calls(self, message: Dict[str, Any], tools: List[Dict[str, Any]], original_prompt: str) -> str:
        """
        Handle tool calls by executing them and getting final response.
        
        Args:
            message: Message containing tool calls
            tools: Available tools
            original_prompt: Original user prompt
            
        Returns:
            Final model response after tool execution
        """
        tool_results = []
        
        # Execute each tool call
        for tool_call in message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            
            # Find the corresponding tool
            tool_result = await self._execute_tool(function_name, arguments, tools)
            
            tool_results.append({
                "role": "tool",
                "content": json.dumps(tool_result),
                "tool_call_id": tool_call.get("id", "unknown")
            })
        
        # Send tool results back to model for final response
        follow_up_payload = {
            "model": self.id,
            "messages": [
                {"role": "user", "content": original_prompt},
                message,  # The assistant's message with tool calls
                *tool_results  # Tool execution results
            ],
            "stream": False
        }
        
        response = await self.client.post(
            f"{self.host}/api/chat",
            json=follow_up_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("message", {}).get("content", "Tool execution completed")
    
    async def _execute_tool(self, function_name: str, arguments: Dict[str, Any], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a specific tool call.
        
        Args:
            function_name: Name of the function to call
            arguments: Function arguments
            tools: Available tools list
            
        Returns:
            Tool execution result
        """
        # Find the tool in our MCP tools
        target_tool = None
        for tool in tools:
            if tool.get("name") == function_name:
                target_tool = tool
                break
        
        if not target_tool:
            return {"error": f"Tool {function_name} not found"}
        
        try:
            # Here you would integrate with your MCP client to actually execute the tool
            # For now, return a placeholder that indicates the tool was called
            logger.info(f"Executing tool: {function_name} with args: {arguments}")
            
            # This is where you'd call your MCP manager
            # result = await self.mcp_manager.call_tool(server_name, function_name, arguments)
            
            return {
                "status": "success",
                "function": function_name,
                "arguments": arguments,
                "result": "Tool executed successfully (placeholder)"
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {e}")
            return {"error": str(e)}
    
    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()


class HybridOllamaAgent:
    """
    Hybrid agent that uses custom Ollama client while maintaining Agno benefits.
    """
    
    def __init__(self, model_id: str, host: str = "http://localhost:11434"):
        """
        Initialize hybrid agent.
        
        Args:
            model_id: Ollama model identifier
            host: Ollama server endpoint
        """
        self.model = CustomOllamaModel(id=model_id, host=host)
        self.tools = []
        self.instructions = []
    
    def add_tools(self, tools: List[Dict[str, Any]]):
        """Add tools to the agent."""
        self.tools = tools
    
    def add_instructions(self, instructions: List[str]):
        """Add instructions to the agent."""
        self.instructions = instructions
    
    async def arun(self, prompt: str) -> str:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Agent response
        """
        # Combine instructions with user prompt
        full_prompt = "\n".join(self.instructions) + "\n\n" + prompt
        
        # Execute with tools
        return await self.model.arun(full_prompt, self.tools)
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.model.aclose()
