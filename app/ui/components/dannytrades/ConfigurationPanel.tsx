/**
 * Configuration Panel Component
 * 
 * Provides configuration interface for LLM providers, MCP servers, and prompts.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { 
  Settings, 
  Brain, 
  Database, 
  Code, 
  Zap, 
  Globe, 
  Key,
  RefreshCw,
  Save,
  Edit
} from 'lucide-react';

interface ConfigurationPanelProps {
  agentStatus: any;
  onStatusChange: (status: any) => void;
}

interface LLMProvider {
  name: string;
  enabled: boolean;
  model_id: string;
  has_api_key: boolean;
  endpoint?: string;
  current: boolean;
}

interface MCPServer {
  name: string;
  enabled: boolean;
  command: string;
  args: string[];
  env_configured: boolean;
}

export const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({ 
  agentStatus, 
  onStatusChange 
}) => {
  const [providers, setProviders] = useState<Record<string, LLMProvider>>({});
  const [mcpServers, setMcpServers] = useState<Record<string, MCPServer>>({});
  const [selectedProvider, setSelectedProvider] = useState<string>('claude');
  const [isLoading, setIsLoading] = useState(false);
  const [mcpConfigOpen, setMcpConfigOpen] = useState(false);
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);

  const [prompts, setPrompts] = useState({
    system_prompt: '',
    processing_instructions: '',
    error_handling_prompt: ''
  });

  const [mcpConfig, setMcpConfig] = useState(`{
  "mcpServers": {
    "notion": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-notion"],
      "env": {
        "NOTION_API_KEY": "\${NOTION_API_KEY}",
        "NOTION_DATABASE_ID": "\${NOTION_DATABASE_ID}"
      }
    }
  }
}`);

  useEffect(() => {
    fetchProviderStatus();
    fetchMcpStatus();
    fetchPrompts();
  }, []);

  const fetchProviderStatus = async () => {
    try {
      const response = await fetch('/api/dannytrades/providers/status');
      if (response.ok) {
        const data = await response.json();
        setProviders(data);
        
        // Find current provider
        const current = Object.entries(data).find(([_, provider]) => provider.current);
        if (current) {
          setSelectedProvider(current[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch provider status:', error);
    }
  };

  const fetchMcpStatus = async () => {
    try {
      const response = await fetch('/api/dannytrades/mcp/status');
      if (response.ok) {
        const data = await response.json();
        setMcpServers(data);
      }
    } catch (error) {
      console.error('Failed to fetch MCP status:', error);
    }
  };

  const fetchPrompts = async () => {
    try {
      const response = await fetch('/api/dannytrades/config');
      if (response.ok) {
        const data = await response.json();
        if (data.prompts) {
          setPrompts(data.prompts);
        }
      }
    } catch (error) {
      console.error('Failed to fetch prompts:', error);
    }
  };

  const fetchOllamaModels = async (endpoint: string) => {
    try {
      const response = await fetch(`${endpoint}/api/tags`);
      if (response.ok) {
        const data = await response.json();
        const models = data.models?.map((model: any) => model.name) || [];
        setOllamaModels(models);
      }
    } catch (error) {
      console.error('Failed to fetch Ollama models:', error);
      setOllamaModels([]);
    }
  };

  const switchProvider = async (providerName: string) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/dannytrades/providers/switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ provider: providerName }),
      });

      if (response.ok) {
        setSelectedProvider(providerName);
        onStatusChange({
          ...agentStatus,
          currentProvider: providerName
        });
        await fetchProviderStatus();
      }
    } catch (error) {
      console.error('Failed to switch provider:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const updatePrompts = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/dannytrades/prompts/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(prompts),
      });

      if (response.ok) {
        setPromptEditorOpen(false);
      }
    } catch (error) {
      console.error('Failed to update prompts:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getProviderIcon = (providerName: string) => {
    switch (providerName) {
      case 'claude':
        return <Brain className="h-4 w-4" />;
      case 'openai':
        return <Zap className="h-4 w-4" />;
      case 'gemini':
        return <Globe className="h-4 w-4" />;
      case 'ollama':
        return <Database className="h-4 w-4" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getProviderStatus = (provider: LLMProvider) => {
    if (!provider.enabled) return { color: 'destructive', text: 'Disabled' };
    if (!provider.has_api_key && provider.name !== 'ollama') return { color: 'secondary', text: 'No API Key' };
    if (provider.current) return { color: 'default', text: 'Active' };
    return { color: 'outline', text: 'Available' };
  };

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* LLM Provider Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>LLM Provider</span>
            </CardTitle>
            <CardDescription>
              Configure and switch between different language model providers
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              {Object.entries(providers).map(([name, provider]) => {
                const status = getProviderStatus(provider);
                return (
                  <div key={name} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getProviderIcon(name)}
                      <div>
                        <p className="font-medium capitalize">{name}</p>
                        <p className="text-sm text-muted-foreground">{provider.model_id}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={status.color as any}>{status.text}</Badge>
                      {provider.enabled && (
                        <Button
                          size="sm"
                          variant={provider.current ? "default" : "outline"}
                          onClick={() => switchProvider(name)}
                          disabled={isLoading || provider.current}
                        >
                          {provider.current ? "Active" : "Select"}
                        </Button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {selectedProvider === 'ollama' && (
              <div className="space-y-2">
                <Label>Ollama Endpoint</Label>
                <div className="flex space-x-2">
                  <Input 
                    placeholder="http://localhost:11434" 
                    defaultValue="http://localhost:11434"
                  />
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => fetchOllamaModels('http://localhost:11434')}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                </div>
                {ollamaModels.length > 0 && (
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select Ollama model" />
                    </SelectTrigger>
                    <SelectContent>
                      {ollamaModels.map((model) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* MCP Server Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Database className="h-5 w-5" />
              <span>MCP Servers</span>
            </CardTitle>
            <CardDescription>
              Model Context Protocol server configurations
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(mcpServers).map(([name, server]) => (
              <div key={name} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <Database className="h-4 w-4" />
                  <div>
                    <p className="font-medium capitalize">{name}</p>
                    <p className="text-sm text-muted-foreground">{server.command}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant={server.enabled ? "default" : "secondary"}>
                    {server.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                  <Badge variant={server.env_configured ? "default" : "destructive"}>
                    {server.env_configured ? "Configured" : "No Config"}
                  </Badge>
                </div>
              </div>
            ))}

            <Dialog open={mcpConfigOpen} onOpenChange={setMcpConfigOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" className="w-full">
                  <Code className="h-4 w-4 mr-2" />
                  Edit MCP Configuration
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>MCP Server Configuration</DialogTitle>
                  <DialogDescription>
                    Edit the JSON configuration for MCP servers
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <Textarea
                    value={mcpConfig}
                    onChange={(e) => setMcpConfig(e.target.value)}
                    className="min-h-[300px] font-mono text-sm"
                    placeholder="Enter MCP configuration JSON..."
                  />
                  <div className="flex justify-end space-x-2">
                    <Button variant="outline" onClick={() => setMcpConfigOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={() => setMcpConfigOpen(false)}>
                      <Save className="h-4 w-4 mr-2" />
                      Save Configuration
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </CardContent>
        </Card>

        {/* Prompt Management */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Edit className="h-5 w-5" />
              <span>Prompt Management</span>
            </CardTitle>
            <CardDescription>
              Customize agent prompts and instructions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Dialog open={promptEditorOpen} onOpenChange={setPromptEditorOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" className="w-full">
                  <Edit className="h-4 w-4 mr-2" />
                  Edit Agent Prompts
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl max-h-[80vh]">
                <DialogHeader>
                  <DialogTitle>Agent Prompt Editor</DialogTitle>
                  <DialogDescription>
                    Customize the prompts used by the DannyTrades agent
                  </DialogDescription>
                </DialogHeader>
                <ScrollArea className="max-h-[60vh]">
                  <div className="space-y-6 pr-4">
                    <div className="space-y-2">
                      <Label>System Prompt</Label>
                      <Textarea
                        value={prompts.system_prompt}
                        onChange={(e) => setPrompts({...prompts, system_prompt: e.target.value})}
                        className="min-h-[120px] font-mono text-sm"
                        placeholder="Enter system prompt..."
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Processing Instructions</Label>
                      <Textarea
                        value={prompts.processing_instructions}
                        onChange={(e) => setPrompts({...prompts, processing_instructions: e.target.value})}
                        className="min-h-[200px] font-mono text-sm"
                        placeholder="Enter processing instructions..."
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Error Handling Prompt</Label>
                      <Textarea
                        value={prompts.error_handling_prompt}
                        onChange={(e) => setPrompts({...prompts, error_handling_prompt: e.target.value})}
                        className="min-h-[100px] font-mono text-sm"
                        placeholder="Enter error handling instructions..."
                      />
                    </div>
                  </div>
                </ScrollArea>
                <div className="flex justify-end space-x-2 pt-4">
                  <Button variant="outline" onClick={() => setPromptEditorOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={updatePrompts} disabled={isLoading}>
                    <Save className="h-4 w-4 mr-2" />
                    {isLoading ? "Saving..." : "Save Prompts"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </CardContent>
        </Card>

        {/* Environment Variables */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Key className="h-5 w-5" />
              <span>Environment</span>
            </CardTitle>
            <CardDescription>
              API keys and environment configuration
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">NOTION_API_KEY</span>
              <Badge variant={mcpServers.notion?.env_configured ? "default" : "destructive"}>
                {mcpServers.notion?.env_configured ? "Configured" : "Missing"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">ANTHROPIC_API_KEY</span>
              <Badge variant={providers.claude?.has_api_key ? "default" : "destructive"}>
                {providers.claude?.has_api_key ? "Configured" : "Missing"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">OPENAI_API_KEY</span>
              <Badge variant={providers.openai?.has_api_key ? "default" : "destructive"}>
                {providers.openai?.has_api_key ? "Configured" : "Missing"}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
};
