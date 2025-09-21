"""
MCP client for connecting to external MCP servers.

This module provides a client for connecting to MCP servers using either
stdio or SSE transport protocols.
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from .config import MCPServerConfig, MCPTransportType

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to MCP servers."""
    
    def __init__(self, server_config: MCPServerConfig):
        """
        Initialize MCP client with server configuration.
        
        Args:
            server_config: Configuration for the MCP server
        """
        self.config = server_config
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []
        self.connected = False
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        try:
            if self.config.transport_type == MCPTransportType.STDIO:
                await self._connect_stdio()
            elif self.config.transport_type == MCPTransportType.SSE:
                await self._connect_sse()
            else:
                raise ValueError(f"Unsupported transport type: {self.config.transport_type}")
            
            # Load available tools
            await self._load_tools()
            self.connected = True
            logger.info(f"Connected to MCP server: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {str(e)}")
            raise
    
    async def _connect_stdio(self) -> None:
        """Connect using stdio transport."""
        if not self.config.command:
            raise ValueError("Command is required for stdio transport")
        
        # Prepare environment variables - merge with current environment
        import os
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)
        
        # Start the MCP server process
        cmd = [self.config.command] + (self.config.args or [])
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        logger.info(f"Started MCP server process: {' '.join(cmd)}")
    
    async def _connect_sse(self) -> None:
        """Connect using SSE transport."""
        if not self.config.url:
            raise ValueError("URL is required for SSE transport")
        
        # TODO: Implement SSE connection
        # This would typically involve setting up an HTTP client
        # to connect to the SSE endpoint
        raise NotImplementedError("SSE transport not yet implemented")
    
    async def _load_tools(self) -> None:
        """Load available tools from the MCP server."""
        if self.config.transport_type == MCPTransportType.STDIO and self.process:
            try:
                # Send tools/list request
                request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                }
                
                # Write request to stdin
                request_json = json.dumps(request) + "\n"
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
                
                # Read response from stdout
                response_line = await self.process.stdout.readline()
                if response_line:
                    response = json.loads(response_line.decode().strip())
                    if "result" in response and "tools" in response["result"]:
                        self.tools = response["result"]["tools"]
                        logger.info(f"Loaded {len(self.tools)} tools from {self.config.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load tools from {self.config.name}: {str(e)}")
                self.tools = []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if not self.connected:
            raise RuntimeError("MCP client not connected")
        
        if self.config.transport_type == MCPTransportType.STDIO and self.process:
            try:
                # Send tools/call request
                request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
                
                # Write request to stdin
                request_json = json.dumps(request) + "\n"
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
                
                # Read response from stdout
                response_line = await self.process.stdout.readline()
                if response_line:
                    response = json.loads(response_line.decode().strip())
                    if "result" in response:
                        return response["result"]
                    elif "error" in response:
                        raise RuntimeError(f"Tool call error: {response['error']}")
                
                raise RuntimeError("No response from MCP server")
                
            except Exception as e:
                logger.error(f"Failed to call tool {tool_name}: {str(e)}")
                raise
        
        raise NotImplementedError(f"Tool calling not implemented for {self.config.transport_type}")
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception as e:
                logger.warning(f"Error terminating MCP server process: {str(e)}")
            finally:
                self.process = None
        
        self.connected = False
        logger.info(f"Disconnected from MCP server: {self.config.name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return self.tools.copy()


class MCPManager:
    """Manager for multiple MCP clients."""
    
    def __init__(self):
        """Initialize MCP manager."""
        self.clients: Dict[str, MCPClient] = {}
    
    async def add_server(self, config: MCPServerConfig) -> None:
        """
        Add and connect to an MCP server.
        
        Args:
            config: Server configuration
        """
        if config.name in self.clients:
            await self.remove_server(config.name)
        
        client = MCPClient(config)
        await client.connect()
        self.clients[config.name] = client
        logger.info(f"Added MCP server: {config.name}")
    
    async def remove_server(self, name: str) -> None:
        """
        Remove and disconnect from an MCP server.
        
        Args:
            name: Server name
        """
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]
            logger.info(f"Removed MCP server: {name}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if server_name not in self.clients:
            raise ValueError(f"MCP server not found: {server_name}")
        
        return await self.clients[server_name].call_tool(tool_name, arguments)
    
    def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available tools from all connected servers.
        
        Returns:
            Dictionary mapping server names to their available tools
        """
        return {
            name: client.get_available_tools()
            for name, client in self.clients.items()
        }
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()
        logger.info("Disconnected from all MCP servers")


@asynccontextmanager
async def mcp_manager(configs: List[MCPServerConfig]):
    """
    Async context manager for MCP servers.
    
    Args:
        configs: List of MCP server configurations
        
    Yields:
        MCPManager instance
    """
    manager = MCPManager()
    
    try:
        # Connect to all configured servers
        for config in configs:
            if config.enabled:
                try:
                    await manager.add_server(config)
                except Exception as e:
                    logger.warning(f"Failed to connect to MCP server {config.name}: {str(e)}")
        
        yield manager
        
    finally:
        # Disconnect from all servers
        await manager.disconnect_all()
