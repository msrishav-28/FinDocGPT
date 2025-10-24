import React, { useState, useEffect, useCallback } from 'react'
import { TrendingUp, TrendingDown, Activity, Wifi, WifiOff } from 'lucide-react'
import { useWebSocket } from '../../hooks'
import AdvancedChart from './AdvancedChart'

const RealTimeMarketData = ({ watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA'], userId = 'demo-user' }) => {
  const [marketData, setMarketData] = useState({})
  const [selectedTicker, setSelectedTicker] = useState(watchlist[0])
  const [historicalData, setHistoricalData] = useState({})
  const [alerts, setAlerts] = useState([])

  // WebSocket connection for real-time updates
  const wsUrl = `ws://localhost:8000/api/ws/connect?user_id=${userId}`
  
  const handleWebSocketMessage = useCallback((message) => {
    switch (message.type) {
      case 'market_data':
        setMarketData(prev => ({
          ...prev,
          [message.data.ticker]: message.data
        }))
        break
      case 'alert':
        setAlerts(prev => [message.data, ...prev.slice(0, 4)]) // Keep last 5 alerts
        break
      case 'connection_established':
        console.log('WebSocket connected:', message.data.message)
        // Subscribe to watchlist tickers
        watchlist.forEach(ticker => {
          sendMessage({
            type: 'subscribe',
            ticker: ticker
          })
        })
        break
      default:
        console.log('Unknown message type:', message.type)
    }
  }, [watchlist])

  const { connectionStatus, sendMessage, readyState } = useWebSocket(wsUrl, {
    onMessage: handleWebSocketMessage,
    maxReconnectAttempts: 5,
    reconnectInterval: 3000
  })

  // Fetch historical data for selected ticker
  useEffect(() => {
    const fetchHistoricalData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/market/historical/${selectedTicker}?period=1d&interval=5m`)
        if (response.ok) {
          const data = await response.json()
          setHistoricalData(prev => ({
            ...prev,
            [selectedTicker]: data
          }))
        }
      } catch (error) {
        console.error('Error fetching historical data:', error)
      }
    }

    if (selectedTicker) {
      fetchHistoricalData()
    }
  }, [selectedTicker])

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price)
  }

  const formatChange = (change, changePercent) => {
    const isPositive = change >= 0
    const sign = isPositive ? '+' : ''
    return {
      change: `${sign}${formatPrice(change)}`,
      changePercent: `${sign}${changePercent?.toFixed(2)}%`,
      isPositive
    }
  }

  const getChartData = (ticker) => {
    const historical = historicalData[ticker]
    if (!historical?.data) return null

    return {
      labels: historical.data.map(point => new Date(point.timestamp)),
      datasets: [
        {
          label: `${ticker} Price`,
          data: historical.data.map(point => point.close),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4
        }
      ]
    }
  }

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center justify-between p-4 rounded-xl glass">
        <div className="flex items-center gap-2">
          {readyState === 1 ? (
            <Wifi className="text-green-400" size={20} />
          ) : (
            <WifiOff className="text-red-400" size={20} />
          )}
          <span className="text-sm text-gray-300">
            Real-time Data: <span className={readyState === 1 ? 'text-green-400' : 'text-red-400'}>
              {connectionStatus}
            </span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Activity className="text-blue-400" size={16} />
          <span className="text-sm text-gray-400">
            {Object.keys(marketData).length} active tickers
          </span>
        </div>
      </div>

      {/* Watchlist Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {watchlist.map(ticker => {
          const data = marketData[ticker]
          const isSelected = ticker === selectedTicker
          
          return (
            <div
              key={ticker}
              onClick={() => setSelectedTicker(ticker)}
              className={`p-4 rounded-xl cursor-pointer transition-all duration-200 ${
                isSelected 
                  ? 'glass border-2 border-blue-500 shadow-lg' 
                  : 'glass hover:border-white/20 border border-white/10'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-white">{ticker}</h3>
                {data && (
                  <div className="flex items-center gap-1">
                    {data.day_change >= 0 ? (
                      <TrendingUp className="text-green-400" size={16} />
                    ) : (
                      <TrendingDown className="text-red-400" size={16} />
                    )}
                  </div>
                )}
              </div>
              
              {data ? (
                <div>
                  <div className="text-xl font-bold text-white mb-1">
                    {formatPrice(data.current_price)}
                  </div>
                  <div className={`text-sm ${data.day_change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatChange(data.day_change, data.day_change_percent).change} 
                    ({formatChange(data.day_change, data.day_change_percent).changePercent})
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Vol: {data.volume?.toLocaleString() || 'N/A'}
                  </div>
                </div>
              ) : (
                <div className="animate-pulse">
                  <div className="h-6 bg-gray-700 rounded mb-2"></div>
                  <div className="h-4 bg-gray-700 rounded w-2/3"></div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Selected Ticker Chart */}
      {selectedTicker && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">
              {selectedTicker} - Intraday Chart
            </h2>
            <div className="text-sm text-gray-400">
              Last updated: {marketData[selectedTicker]?.timestamp ? 
                new Date(marketData[selectedTicker].timestamp).toLocaleTimeString() : 
                'N/A'
              }
            </div>
          </div>
          
          <AdvancedChart
            data={getChartData(selectedTicker)}
            title={`${selectedTicker} Price Movement`}
            height={400}
            realTimeData={null} // Real-time updates are merged in the chart component
            zoomEnabled={true}
            onDataPointClick={(point) => {
              console.log('Chart point clicked:', point)
            }}
          />
        </div>
      )}

      {/* Recent Alerts */}
      {alerts.length > 0 && (
        <div className="p-4 rounded-xl glass">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Recent Alerts</h3>
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <div key={index} className="flex items-center gap-2 p-2 rounded-md bg-yellow-900/20 border border-yellow-900/30">
                <Activity className="text-yellow-400" size={16} />
                <span className="text-sm text-yellow-200">{alert.message}</span>
                <span className="text-xs text-gray-400 ml-auto">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default RealTimeMarketData