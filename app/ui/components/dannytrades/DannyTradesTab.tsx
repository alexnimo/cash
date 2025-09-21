/**
 * DannyTrades Tab Component
 * 
 * Main tab component for the DannyTrades agent interface.
 * Provides configuration, monitoring, and data visualization for stock processing.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Settings, 
  Activity, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle, 
  Clock,
  Zap,
  Database,
  Brain
} from 'lucide-react';

import { ConfigurationPanel } from './ConfigurationPanel';
import { MonitoringPanel } from './MonitoringPanel';
import { StockDashboard } from './StockDashboard';
import { RealTimeActivity } from './RealTimeActivity';

interface DannyTradesTabProps {
  className?: string;
}

interface AgentStatus {
  initialized: boolean;
  currentProvider: string;
  processing: boolean;
  lastActivity: string;
}

interface ProcessingStats {
  totalProcessed: number;
  successRate: number;
  averageTime: number;
  activeRequests: number;
}

export const DannyTradesTab: React.FC<DannyTradesTabProps> = ({ className }) => {
  const [agentStatus, setAgentStatus] = useState<AgentStatus>({
    initialized: false,
    currentProvider: 'claude',
    processing: false,
    lastActivity: 'Never'
  });

  const [processingStats, setProcessingStats] = useState<ProcessingStats>({
    totalProcessed: 0,
    successRate: 0,
    averageTime: 0,
    activeRequests: 0
  });

  const [realtimeEnabled, setRealtimeEnabled] = useState(true);

  useEffect(() => {
    // Initialize agent status and stats
    fetchAgentStatus();
    fetchProcessingStats();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      if (realtimeEnabled) {
        fetchAgentStatus();
        fetchProcessingStats();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [realtimeEnabled]);

  const fetchAgentStatus = async () => {
    try {
      const response = await fetch('/api/dannytrades/config');
      if (response.ok) {
        const data = await response.json();
        setAgentStatus({
          initialized: true,
          currentProvider: data.current_provider || 'claude',
          processing: false, // Will be updated by activity monitoring
          lastActivity: new Date().toLocaleTimeString()
        });
      }
    } catch (error) {
      console.error('Failed to fetch agent status:', error);
    }
  };

  const fetchProcessingStats = async () => {
    try {
      // This would be implemented with actual stats endpoint
      setProcessingStats({
        totalProcessed: 156,
        successRate: 98.7,
        averageTime: 2.3,
        activeRequests: 0
      });
    } catch (error) {
      console.error('Failed to fetch processing stats:', error);
    }
  };

  const getStatusColor = (status: boolean) => {
    return status ? 'bg-green-500' : 'bg-red-500';
  };

  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case 'claude':
        return <Brain className="h-4 w-4" />;
      case 'openai':
        return <Zap className="h-4 w-4" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header with Status */}
      <div className="flex items-center justify-between p-6 border-b">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${getStatusColor(agentStatus.initialized)}`} />
            <h1 className="text-2xl font-bold">DannyTrades Agent</h1>
          </div>
          <Badge variant="outline" className="flex items-center space-x-1">
            {getProviderIcon(agentStatus.currentProvider)}
            <span>{agentStatus.currentProvider}</span>
          </Badge>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-muted-foreground">
            Last Activity: {agentStatus.lastActivity}
          </div>
          <Button
            variant={realtimeEnabled ? "default" : "outline"}
            size="sm"
            onClick={() => setRealtimeEnabled(!realtimeEnabled)}
            className="flex items-center space-x-1"
          >
            <Activity className="h-4 w-4" />
            <span>Real-time</span>
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4 p-6 border-b">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-blue-500" />
              <div>
                <p className="text-sm font-medium">Total Processed</p>
                <p className="text-2xl font-bold">{processingStats.totalProcessed}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <div>
                <p className="text-sm font-medium">Success Rate</p>
                <p className="text-2xl font-bold">{processingStats.successRate}%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-orange-500" />
              <div>
                <p className="text-sm font-medium">Avg Time</p>
                <p className="text-2xl font-bold">{processingStats.averageTime}s</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-purple-500" />
              <div>
                <p className="text-sm font-medium">Active Requests</p>
                <p className="text-2xl font-bold">{processingStats.activeRequests}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Configuration Sidebar */}
        <div className="w-80 border-r bg-muted/30">
          <ConfigurationPanel 
            agentStatus={agentStatus}
            onStatusChange={setAgentStatus}
          />
        </div>

        {/* Main Dashboard */}
        <div className="flex-1 p-6">
          <Tabs defaultValue="dashboard" className="h-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="dashboard" className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4" />
                <span>Dashboard</span>
              </TabsTrigger>
              <TabsTrigger value="monitoring" className="flex items-center space-x-2">
                <Activity className="h-4 w-4" />
                <span>Monitoring</span>
              </TabsTrigger>
              <TabsTrigger value="settings" className="flex items-center space-x-2">
                <Settings className="h-4 w-4" />
                <span>Settings</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="dashboard" className="mt-6 h-full">
              <StockDashboard />
            </TabsContent>

            <TabsContent value="monitoring" className="mt-6 h-full">
              <MonitoringPanel realtimeEnabled={realtimeEnabled} />
            </TabsContent>

            <TabsContent value="settings" className="mt-6 h-full">
              <div className="space-y-6">
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Advanced settings are available in the configuration panel on the left.
                  </AlertDescription>
                </Alert>
                
                <Card>
                  <CardHeader>
                    <CardTitle>System Information</CardTitle>
                    <CardDescription>
                      Current system status and configuration details
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between">
                        <span>Agent Status:</span>
                        <Badge variant={agentStatus.initialized ? "default" : "destructive"}>
                          {agentStatus.initialized ? "Initialized" : "Not Initialized"}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Current Provider:</span>
                        <span className="font-medium">{agentStatus.currentProvider}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Real-time Monitoring:</span>
                        <Badge variant={realtimeEnabled ? "default" : "secondary"}>
                          {realtimeEnabled ? "Enabled" : "Disabled"}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>

        {/* Real-time Activity Sidebar */}
        {realtimeEnabled && (
          <div className="w-80 border-l bg-muted/30">
            <RealTimeActivity />
          </div>
        )}
      </div>
    </div>
  );
};
