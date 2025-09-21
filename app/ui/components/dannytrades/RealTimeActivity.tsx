/**
 * Real-time Activity Component
 * 
 * Sidebar component for displaying live agent activities and notifications.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  Activity, 
  Bell, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Zap,
  TrendingUp,
  Database,
  RefreshCw,
  X
} from 'lucide-react';

interface RealTimeActivityProps {
  className?: string;
}

interface LiveActivity {
  id: string;
  timestamp: string;
  type: 'processing' | 'success' | 'error' | 'notification';
  title: string;
  description?: string;
  stock_symbol?: string;
  dismissible?: boolean;
}

interface LiveStats {
  processing_count: number;
  success_count: number;
  error_count: number;
  last_update: string;
}

export const RealTimeActivity: React.FC<RealTimeActivityProps> = ({ className }) => {
  const [activities, setActivities] = useState<LiveActivity[]>([]);
  const [stats, setStats] = useState<LiveStats>({
    processing_count: 0,
    success_count: 0,
    error_count: 0,
    last_update: new Date().toISOString()
  });
  const [notifications, setNotifications] = useState<LiveActivity[]>([]);

  // Mock real-time data
  useEffect(() => {
    // Initialize with some activities
    const initialActivities: LiveActivity[] = [
      {
        id: '1',
        timestamp: new Date().toISOString(),
        type: 'processing',
        title: 'Processing AAPL data',
        description: 'Analyzing technical indicators',
        stock_symbol: 'AAPL'
      },
      {
        id: '2',
        timestamp: new Date(Date.now() - 30000).toISOString(),
        type: 'success',
        title: 'TSLA updated successfully',
        description: 'Notion database entry created',
        stock_symbol: 'TSLA'
      }
    ];

    const initialNotifications: LiveActivity[] = [
      {
        id: 'n1',
        timestamp: new Date().toISOString(),
        type: 'notification',
        title: 'High volatility detected',
        description: 'NVDA showing unusual price movement',
        stock_symbol: 'NVDA',
        dismissible: true
      }
    ];

    setActivities(initialActivities);
    setNotifications(initialNotifications);

    // Simulate real-time updates
    const interval = setInterval(() => {
      // Update stats
      setStats(prev => ({
        ...prev,
        processing_count: Math.floor(Math.random() * 3),
        success_count: prev.success_count + Math.floor(Math.random() * 2),
        error_count: prev.error_count + (Math.random() < 0.1 ? 1 : 0),
        last_update: new Date().toISOString()
      }));

      // Occasionally add new activity
      if (Math.random() < 0.4) {
        const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'];
        const types: LiveActivity['type'][] = ['processing', 'success', 'error'];
        const titles = {
          processing: 'Processing data',
          success: 'Update completed',
          error: 'Processing failed'
        };

        const newActivity: LiveActivity = {
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          type: types[Math.floor(Math.random() * types.length)],
          title: titles[types[Math.floor(Math.random() * types.length)]],
          stock_symbol: symbols[Math.floor(Math.random() * symbols.length)]
        };

        setActivities(prev => [newActivity, ...prev.slice(0, 19)]);
      }

      // Occasionally add notification
      if (Math.random() < 0.1) {
        const newNotification: LiveActivity = {
          id: `n${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: 'notification',
          title: 'Market alert',
          description: 'Unusual activity detected',
          dismissible: true
        };

        setNotifications(prev => [newNotification, ...prev]);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getActivityIcon = (type: LiveActivity['type']) => {
    switch (type) {
      case 'processing':
        return <RefreshCw className="h-3 w-3 text-blue-500 animate-spin" />;
      case 'success':
        return <CheckCircle className="h-3 w-3 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-3 w-3 text-red-500" />;
      case 'notification':
        return <Bell className="h-3 w-3 text-yellow-500" />;
      default:
        return <Activity className="h-3 w-3 text-gray-500" />;
    }
  };

  const getActivityColor = (type: LiveActivity['type']) => {
    switch (type) {
      case 'processing':
        return 'border-l-blue-500';
      case 'success':
        return 'border-l-green-500';
      case 'error':
        return 'border-l-red-500';
      case 'notification':
        return 'border-l-yellow-500';
      default:
        return 'border-l-gray-500';
    }
  };

  const dismissNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className={`h-full flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center space-x-2 mb-3">
          <Activity className="h-5 w-5" />
          <h3 className="font-semibold">Live Activity</h3>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 bg-blue-50 rounded">
            <p className="text-lg font-bold text-blue-600">{stats.processing_count}</p>
            <p className="text-xs text-blue-600">Processing</p>
          </div>
          <div className="p-2 bg-green-50 rounded">
            <p className="text-lg font-bold text-green-600">{stats.success_count}</p>
            <p className="text-xs text-green-600">Success</p>
          </div>
          <div className="p-2 bg-red-50 rounded">
            <p className="text-lg font-bold text-red-600">{stats.error_count}</p>
            <p className="text-xs text-red-600">Errors</p>
          </div>
        </div>
      </div>

      {/* Notifications */}
      {notifications.length > 0 && (
        <div className="p-4 border-b bg-yellow-50/50">
          <div className="flex items-center space-x-2 mb-2">
            <Bell className="h-4 w-4 text-yellow-600" />
            <span className="text-sm font-medium text-yellow-800">Notifications</span>
          </div>
          <div className="space-y-2">
            {notifications.slice(0, 3).map((notification) => (
              <div key={notification.id} className="flex items-start justify-between p-2 bg-white rounded border">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{notification.title}</p>
                  {notification.description && (
                    <p className="text-xs text-muted-foreground truncate">{notification.description}</p>
                  )}
                  {notification.stock_symbol && (
                    <Badge variant="outline" className="text-xs mt-1">
                      {notification.stock_symbol}
                    </Badge>
                  )}
                </div>
                {notification.dismissible && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0"
                    onClick={() => dismissNotification(notification.id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Activity Stream */}
      <div className="flex-1 overflow-hidden">
        <div className="p-4 pb-2">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium">Recent Activity</span>
            <span className="text-xs text-muted-foreground">
              {formatTime(stats.last_update)}
            </span>
          </div>
        </div>
        
        <ScrollArea className="flex-1 px-4">
          <div className="space-y-2 pb-4">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className={`flex items-start space-x-2 p-2 border-l-2 ${getActivityColor(activity.type)} bg-muted/30 rounded-r`}
              >
                <div className="mt-1">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium truncate">{activity.title}</p>
                    <span className="text-xs text-muted-foreground">
                      {formatTime(activity.timestamp)}
                    </span>
                  </div>
                  {activity.description && (
                    <p className="text-xs text-muted-foreground truncate mt-0.5">
                      {activity.description}
                    </p>
                  )}
                  {activity.stock_symbol && (
                    <Badge variant="outline" className="text-xs mt-1">
                      {activity.stock_symbol}
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Footer */}
      <div className="p-4 border-t bg-muted/30">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Auto-refresh: 3s</span>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span>Live</span>
          </div>
        </div>
      </div>
    </div>
  );
};
