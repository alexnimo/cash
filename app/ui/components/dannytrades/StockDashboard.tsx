/**
 * Stock Dashboard Component
 * 
 * Main dashboard for displaying updated stock information, hot stocks, and data visualization.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { 
  TrendingUp, 
  TrendingDown, 
  Search, 
  Filter, 
  Eye,
  RefreshCw,
  Calendar,
  DollarSign,
  BarChart3,
  Activity
} from 'lucide-react';

interface StockData {
  symbol: string;
  ticker_name?: string;
  last_updated: string;
  volatility_hole: boolean;
  red_candle: {
    daily: boolean;
    weekly: boolean;
    monthly: boolean;
  };
  yellow_candle: {
    daily: boolean;
    weekly: boolean;
    monthly: boolean;
  };
  whale_accumulation: {
    daily: string;
    weekly: string;
    monthly: string;
  };
  technical_indicators: {
    support_levels: number[];
    resistance_levels: number[];
  };
  bullishness_signal: string[];
  misc_data?: string;
  patreon_link?: string;
}

interface StockDashboardProps {
  className?: string;
}

export const StockDashboard: React.FC<StockDashboardProps> = ({ className }) => {
  const [stocks, setStocks] = useState<StockData[]>([]);
  const [filteredStocks, setFilteredStocks] = useState<StockData[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Mock data - in production, this would come from the API
  const mockStocks: StockData[] = [
    {
      symbol: 'AAPL',
      ticker_name: 'Apple Inc.',
      last_updated: '2025-01-07T20:30:00Z',
      volatility_hole: true,
      red_candle: { daily: true, weekly: false, monthly: true },
      yellow_candle: { daily: false, weekly: true, monthly: false },
      whale_accumulation: { daily: 'Strong', weekly: 'Moderate', monthly: 'Weak' },
      technical_indicators: { support_levels: [145, 140, 135], resistance_levels: [155, 160, 165] },
      bullishness_signal: ['Extreme Bullish'],
      misc_data: 'Strong institutional buying detected with high volume',
      patreon_link: 'https://patreon.com/dannytrades'
    },
    {
      symbol: 'TSLA',
      ticker_name: 'Tesla Inc.',
      last_updated: '2025-01-07T20:25:00Z',
      volatility_hole: false,
      red_candle: { daily: false, weekly: true, monthly: false },
      yellow_candle: { daily: true, weekly: false, monthly: true },
      whale_accumulation: { daily: 'Moderate', weekly: 'Strong', monthly: 'Strong' },
      technical_indicators: { support_levels: [240, 235, 230], resistance_levels: [260, 270, 280] },
      bullishness_signal: ['Hot'],
      misc_data: 'Consolidation pattern forming with potential breakout'
    },
    {
      symbol: 'NVDA',
      ticker_name: 'NVIDIA Corporation',
      last_updated: '2025-01-07T20:20:00Z',
      volatility_hole: true,
      red_candle: { daily: true, weekly: true, monthly: false },
      yellow_candle: { daily: false, weekly: false, monthly: true },
      whale_accumulation: { daily: 'Strong', weekly: 'Strong', monthly: 'Moderate' },
      technical_indicators: { support_levels: [880, 870, 860], resistance_levels: [920, 940, 960] },
      bullishness_signal: ['Extreme Bullish', 'Hot'],
      misc_data: 'AI sector momentum continues with strong fundamentals'
    }
  ];

  useEffect(() => {
    // Initialize with mock data
    setStocks(mockStocks);
    setFilteredStocks(mockStocks);
  }, []);

  useEffect(() => {
    // Filter stocks based on search and filter criteria
    let filtered = stocks;

    if (searchTerm) {
      filtered = filtered.filter(stock => 
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.ticker_name?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterType !== 'all') {
      switch (filterType) {
        case 'hot':
          filtered = filtered.filter(stock => stock.bullishness_signal.includes('Hot'));
          break;
        case 'extreme_bullish':
          filtered = filtered.filter(stock => stock.bullishness_signal.includes('Extreme Bullish'));
          break;
        case 'volatility_hole':
          filtered = filtered.filter(stock => stock.volatility_hole);
          break;
        case 'red_candle':
          filtered = filtered.filter(stock => 
            stock.red_candle.daily || stock.red_candle.weekly || stock.red_candle.monthly
          );
          break;
      }
    }

    setFilteredStocks(filtered);
  }, [stocks, searchTerm, filterType]);

  const refreshData = async () => {
    setIsLoading(true);
    try {
      // In production, fetch from API
      // const response = await fetch('/api/dannytrades/stocks');
      // const data = await response.json();
      // setStocks(data);
      
      // For now, simulate refresh
      await new Promise(resolve => setTimeout(resolve, 1000));
      setLastRefresh(new Date());
    } catch (error) {
      console.error('Failed to refresh stock data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getBullishnessColor = (signals: string[]) => {
    if (signals.includes('Extreme Bullish')) return 'bg-green-500';
    if (signals.includes('Hot')) return 'bg-orange-500';
    return 'bg-gray-500';
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Search and Filters */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search stocks..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-64"
            />
          </div>
          
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="w-48">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Stocks</SelectItem>
              <SelectItem value="hot">Hot Stocks</SelectItem>
              <SelectItem value="extreme_bullish">Extreme Bullish</SelectItem>
              <SelectItem value="volatility_hole">Volatility Hole</SelectItem>
              <SelectItem value="red_candle">Red Candle</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-sm text-muted-foreground">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </span>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={refreshData}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4 text-blue-500" />
              <div>
                <p className="text-sm font-medium">Total Updated</p>
                <p className="text-2xl font-bold">{stocks.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <div>
                <p className="text-sm font-medium">Extreme Bullish</p>
                <p className="text-2xl font-bold">
                  {stocks.filter(s => s.bullishness_signal.includes('Extreme Bullish')).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-orange-500" />
              <div>
                <p className="text-sm font-medium">Hot Stocks</p>
                <p className="text-2xl font-bold">
                  {stocks.filter(s => s.bullishness_signal.includes('Hot')).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-4 w-4 text-purple-500" />
              <div>
                <p className="text-sm font-medium">Volatility Holes</p>
                <p className="text-2xl font-bold">
                  {stocks.filter(s => s.volatility_hole).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stock List */}
      <Card>
        <CardHeader>
          <CardTitle>Recently Updated Stocks</CardTitle>
          <CardDescription>
            Showing {filteredStocks.length} of {stocks.length} stocks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <div className="space-y-3">
              {filteredStocks.map((stock) => (
                <div
                  key={stock.symbol}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 cursor-pointer"
                  onClick={() => setSelectedStock(stock)}
                >
                  <div className="flex items-center space-x-4">
                    <div className={`w-3 h-3 rounded-full ${getBullishnessColor(stock.bullishness_signal)}`} />
                    <div>
                      <div className="flex items-center space-x-2">
                        <p className="font-semibold">{stock.symbol}</p>
                        {stock.ticker_name && (
                          <p className="text-sm text-muted-foreground">{stock.ticker_name}</p>
                        )}
                      </div>
                      <div className="flex items-center space-x-2 mt-1">
                        {stock.bullishness_signal.map((signal) => (
                          <Badge key={signal} variant="secondary" className="text-xs">
                            {signal}
                          </Badge>
                        ))}
                        {stock.volatility_hole && (
                          <Badge variant="outline" className="text-xs">
                            Volatility Hole
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        Support: {stock.technical_indicators.support_levels.slice(0, 2).join(', ')}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Resistance: {stock.technical_indicators.resistance_levels.slice(0, 2).join(', ')}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">
                        {formatTimeAgo(stock.last_updated)}
                      </p>
                      <Button variant="ghost" size="sm">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Stock Detail Modal */}
      <Dialog open={!!selectedStock} onOpenChange={() => setSelectedStock(null)}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          {selectedStock && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center space-x-2">
                  <span>{selectedStock.symbol}</span>
                  {selectedStock.ticker_name && (
                    <span className="text-muted-foreground">- {selectedStock.ticker_name}</span>
                  )}
                </DialogTitle>
                <DialogDescription>
                  Last updated: {new Date(selectedStock.last_updated).toLocaleString()}
                </DialogDescription>
              </DialogHeader>

              <ScrollArea className="max-h-[60vh]">
                <div className="space-y-6 pr-4">
                  {/* Bullishness Signals */}
                  <div>
                    <h3 className="font-semibold mb-2">Bullishness Signals</h3>
                    <div className="flex space-x-2">
                      {selectedStock.bullishness_signal.map((signal) => (
                        <Badge key={signal} className={getBullishnessColor([signal])}>
                          {signal}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Technical Indicators */}
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Support Levels</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-1">
                          {selectedStock.technical_indicators.support_levels.map((level, index) => (
                            <div key={index} className="text-sm">
                              ${level.toFixed(2)}
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Resistance Levels</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-1">
                          {selectedStock.technical_indicators.resistance_levels.map((level, index) => (
                            <div key={index} className="text-sm">
                              ${level.toFixed(2)}
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Candle Analysis */}
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Red Candles</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span>Daily:</span>
                            <Badge variant={selectedStock.red_candle.daily ? "destructive" : "outline"}>
                              {selectedStock.red_candle.daily ? "Yes" : "No"}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span>Weekly:</span>
                            <Badge variant={selectedStock.red_candle.weekly ? "destructive" : "outline"}>
                              {selectedStock.red_candle.weekly ? "Yes" : "No"}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span>Monthly:</span>
                            <Badge variant={selectedStock.red_candle.monthly ? "destructive" : "outline"}>
                              {selectedStock.red_candle.monthly ? "Yes" : "No"}
                            </Badge>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Yellow Candles</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span>Daily:</span>
                            <Badge variant={selectedStock.yellow_candle.daily ? "default" : "outline"}>
                              {selectedStock.yellow_candle.daily ? "Yes" : "No"}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span>Weekly:</span>
                            <Badge variant={selectedStock.yellow_candle.weekly ? "default" : "outline"}>
                              {selectedStock.yellow_candle.weekly ? "Yes" : "No"}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span>Monthly:</span>
                            <Badge variant={selectedStock.yellow_candle.monthly ? "default" : "outline"}>
                              {selectedStock.yellow_candle.monthly ? "Yes" : "No"}
                            </Badge>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Whale Accumulation */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Whale Accumulation Analysis</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>Daily:</span>
                          <span className="font-medium">{selectedStock.whale_accumulation.daily}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Weekly:</span>
                          <span className="font-medium">{selectedStock.whale_accumulation.weekly}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Monthly:</span>
                          <span className="font-medium">{selectedStock.whale_accumulation.monthly}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Additional Details */}
                  {selectedStock.misc_data && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Additional Analysis</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm">{selectedStock.misc_data}</p>
                      </CardContent>
                    </Card>
                  )}

                  {/* Chart Placeholder */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Price Chart</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 bg-muted rounded-lg flex items-center justify-center">
                        <div className="text-center text-muted-foreground">
                          <BarChart3 className="h-12 w-12 mx-auto mb-2" />
                          <p>TradingView Chart Integration</p>
                          <p className="text-xs">Coming Soon</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};
