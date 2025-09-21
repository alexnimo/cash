/**
 * Monitoring Panel Component
 * 
 * Real-time monitoring interface for agent activities, performance metrics, and system health.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Zap,
  Database,
  TrendingUp,
  AlertTriangle,
  Info,
  RefreshCw
} from 'lucide-react';

interface MonitoringPanelProps {
  realtimeEnabled: boolean;
  className?: string;
}

interface ActivityLog {
  id: string;
  timestamp: string;
  type: 'processing' | 'success' | 'error' | 'warning' | 'info';
  message: string;
  details?: string;
  duration?: number;
  stock_symbol?: string;
}

interface PerformanceMetrics {
  cpu_usage: number;
  memory_usage: number;
  active_connections: number;
  queue_size: number;
  response_time_avg: number;
  error_rate: number;
}

interface SystemHealth {
  agent_status: 'healthy' | 'warning' | 'error';
  mcp_status: 'connected' | 'disconnected' | 'error';
  notion_status: 'connected' | 'disconnected' | 'error';
  llm_status: 'available' | 'rate_limited' | 'error';
}

export const MonitoringPanel: React.FC<MonitoringPanelProps> = ({ 
  realtimeEnabled, 
  className 
}) => {
  const [activities, setActivities] = useState<ActivityLog[]>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    cpu_usage: 0,
    memory_usage: 0,
    active_connections: 0,
    queue_size: 0,
    response_time_avg: 0,
    error_rate: 0
  });
  const [health, setHealth] = useState<SystemHealth>({
    agent_status: 'healthy',
    mcp_status: 'connected',
    notion_status: 'connected',
    llm_status: 'available'
  });

  // Mock data for demonstration
  const mockActivities: ActivityLog[] = [
    {
      id: '1',
      timestamp: new Date(Date.now() - 30000).toISOString(),
      type: 'success',
      message: 'Successfully processed stock data for AAPL',
      stock_symbol: 'AAPL',
      duration: 2.3
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 120000).toISOString(),
      type: 'processing',
      message: 'Processing batch of 5 stock updates',
      details: 'TSLA, NVDA, MSFT, GOOGL, AMZN'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 180000).toISOString(),
      type: 'warning',
      message: 'Rate limit approaching for Anthropic API',
      details: 'Current usage: 85% of daily limit'
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 240000).toISOString(),
      type: 'success',
      message: 'Notion database updated successfully',
      details: '3 new entries, 2 updates'
    },
    {
      id: '5',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      type: 'error',
      message: 'Failed to process TSLA data',
      details: 'Invalid JSON structure in input data'
    }
  ];

  useEffect(() => {
    // Initialize with mock data
    setActivities(mockActivities);
    
    // Simulate real-time updates
    if (realtimeEnabled) {
      const interval = setInterval(() => {
        // Update metrics
        setMetrics({
          cpu_usage: Math.random() * 100,
          memory_usage: 60 + Math.random() * 20,
          active_connections: Math.floor(Math.random() * 10),
          queue_size: Math.floor(Math.random() * 5),
          response_time_avg: 1.5 + Math.random() * 2,
          error_rate: Math.random() * 5
        });

        // Occasionally add new activity
        if (Math.random() < 0.3) {
          const newActivity: ActivityLog = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            type: ['success', 'processing', 'info'][Math.floor(Math.random() * 3)] as any,
            message: 'New activity logged',
            stock_symbol: ['AAPL', 'TSLA', 'NVDA'][Math.floor(Math.random() * 3)]
          };
          
          setActivities(prev => [newActivity, ...prev.slice(0, 49)]);
        }
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [realtimeEnabled]);

  const getActivityIcon = (type: ActivityLog['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'processing':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Info className="h-4 w-4 text-gray-500" />;
    }
  };

  const getActivityBadgeVariant = (type: ActivityLog['type']) => {
    switch (type) {
      case 'success':
        return 'default';
      case 'error':
        return 'destructive';
      case 'warning':
        return 'secondary';
      case 'processing':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const getHealthStatus = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
      case 'available':
        return { color: 'bg-green-500', text: 'Healthy' };
      case 'warning':
      case 'rate_limited':
        return { color: 'bg-yellow-500', text: 'Warning' };
      case 'error':
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Error' };
      default:
        return { color: 'bg-gray-500', text: 'Unknown' };
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    
    if (diffSecs < 60) return `${diffSecs}s ago`;
    const diffMins = Math.floor(diffSecs / 60);
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    return `${diffHours}h ago`;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* System Health Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>System Health</span>
          </CardTitle>
          <CardDescription>
            Real-time status of all system components
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Agent Status</span>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${getHealthStatus(health.agent_status).color}`} />
                <span className="text-sm font-medium">{getHealthStatus(health.agent_status).text}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm">MCP Connection</span>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${getHealthStatus(health.mcp_status).color}`} />
                <span className="text-sm font-medium">{getHealthStatus(health.mcp_status).text}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm">Notion API</span>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${getHealthStatus(health.notion_status).color}`} />
                <span className="text-sm font-medium">{getHealthStatus(health.notion_status).text}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm">LLM Provider</span>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${getHealthStatus(health.llm_status).color}`} />
                <span className="text-sm font-medium">{getHealthStatus(health.llm_status).text}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Performance Metrics</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>CPU Usage</span>
                <span>{metrics.cpu_usage.toFixed(1)}%</span>
              </div>
              <Progress value={metrics.cpu_usage} className="h-2" />
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Memory Usage</span>
                <span>{metrics.memory_usage.toFixed(1)}%</span>
              </div>
              <Progress value={metrics.memory_usage} className="h-2" />
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="text-center">
                <p className="text-2xl font-bold">{metrics.active_connections}</p>
                <p className="text-sm text-muted-foreground">Active Connections</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{metrics.queue_size}</p>
                <p className="text-sm text-muted-foreground">Queue Size</p>
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <p className="text-lg font-semibold">{metrics.response_time_avg.toFixed(2)}s</p>
                <p className="text-sm text-muted-foreground">Avg Response Time</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-semibold">{metrics.error_rate.toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground">Error Rate</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Activity Log */}
      <Card className="flex-1">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Clock className="h-5 w-5" />
              <span>Activity Log</span>
            </CardTitle>
            <Badge variant={realtimeEnabled ? "default" : "secondary"}>
              {realtimeEnabled ? "Live" : "Paused"}
            </Badge>
          </div>
          <CardDescription>
            Real-time agent activities and system events
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <div className="space-y-3">
              {activities.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3 p-3 border rounded-lg">
                  <div className="mt-0.5">
                    {getActivityIcon(activity.type)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-medium truncate">{activity.message}</p>
                      <div className="flex items-center space-x-2">
                        {activity.stock_symbol && (
                          <Badge variant="outline" className="text-xs">
                            {activity.stock_symbol}
                          </Badge>
                        )}
                        <Badge variant={getActivityBadgeVariant(activity.type)} className="text-xs">
                          {activity.type}
                        </Badge>
                      </div>
                    </div>
                    
                    {activity.details && (
                      <p className="text-xs text-muted-foreground mb-1">{activity.details}</p>
                    )}
                    
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{formatTimeAgo(activity.timestamp)}</span>
                      {activity.duration && (
                        <span>{activity.duration}s</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
};
