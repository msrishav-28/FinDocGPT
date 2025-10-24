import React, { useState, useEffect, useCallback } from 'react'
import { Plus, X, Star, TrendingUp, TrendingDown, AlertTriangle, Settings } from 'lucide-react'
import { AdvancedChart } from '../organisms'

const WatchlistManager = ({ userId = 'demo-user', onTickerSelect }) => {
  const [watchlists, setWatchlists] = useState([
    { id: 'default', name: 'My Watchlist', tickers: ['AAPL', 'MSFT', 'GOOGL', 'TSLA'] },
    { id: 'tech', name: 'Tech Stocks', tickers: ['NVDA', 'META', 'NFLX', 'AMZN'] }
  ])
  const [activeWatchlist, setActiveWatchlist] = useState('default')
  const [newTicker, setNewTicker] = useState('')
  const [newWatchlistName, setNewWatchlistName] = useState('')
  const [showAddWatchlist, setShowAddWatchlist] = useState(false)
  const [marketData, setMarketData] = useState({})
  const [loading, setLoading] = useState(false)

  // Fetch market data for watchlist tickers
  const fetchMarketData = useCallback(async (tickers) => {
    setLoading(true)
    try {
      const promises = tickers.map(async (ticker) => {
        const response = await fetch(`http://localhost:8000/api/market/quote/${ticker}`)
        if (response.ok) {
          const data = await response.json()
          return { ticker, data }
        }
        return { ticker, data: null }
      })
      
      const results = await Promise.all(promises)
      const dataMap = {}
      results.forEach(({ ticker, data }) => {
        if (data) {
          dataMap[ticker] = data
        }
      })
      
      setMarketData(prev => ({ ...prev, ...dataMap }))
    } catch (error) {
      console.error('Error fetching market data:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  // Load market data when active watchlist changes
  useEffect(() => {
    const currentWatchlist = watchlists.find(w => w.id === activeWatchlist)
    if (currentWatchlist) {
      fetchMarketData(currentWatchlist.tickers)
    }
  }, [activeWatchlist, watchlists, fetchMarketData])

  const addTicker = (watchlistId, ticker) => {
    const upperTicker = ticker.toUpperCase()
    setWatchlists(prev => prev.map(w => 
      w.id === watchlistId 
        ? { ...w, tickers: [...new Set([...w.tickers, upperTicker])] }
        : w
    ))
    setNewTicker('')
  }

  const removeTicker = (watchlistId, ticker) => {
    setWatchlists(prev => prev.map(w => 
      w.id === watchlistId 
        ? { ...w, tickers: w.tickers.filter(t => t !== ticker) }
        : w
    ))
  }

  const createWatchlist = (name) => {
    const id = name.toLowerCase().replace(/\s+/g, '-')
    setWatchlists(prev => [...prev, { id, name, tickers: [] }])
    setNewWatchlistName('')
    setShowAddWatchlist(false)
    setActiveWatchlist(id)
  }

  const deleteWatchlist = (watchlistId) => {
    if (watchlists.length <= 1) return // Keep at least one watchlist
    setWatchlists(prev => prev.filter(w => w.id !== watchlistId))
    if (activeWatchlist === watchlistId) {
      setActiveWatchlist(watchlists[0].id)
    }
  }

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

  const currentWatchlist = watchlists.find(w => w.id === activeWatchlist)

  return (
    <div className="space-y-6">
      {/* Watchlist Tabs */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 overflow-x-auto">
          {watchlists.map(watchlist => (
            <button
              key={watchlist.id}
              onClick={() => setActiveWatchlist(watchlist.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                activeWatchlist === watchlist.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {watchlist.name}
              <span className="ml-2 text-xs opacity-75">
                ({watchlist.tickers.length})
              </span>
            </button>
          ))}
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAddWatchlist(true)}
            className="p-2 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors"
            title="Create new watchlist"
          >
            <Plus size={16} />
          </button>
          <button
            onClick={() => currentWatchlist && fetchMarketData(currentWatchlist.tickers)}
            disabled={loading}
            className="p-2 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50"
            title="Refresh data"
          >
            <Settings size={16} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Add New Watchlist Modal */}
      {showAddWatchlist && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Create New Watchlist</h3>
            <input
              type="text"
              value={newWatchlistName}
              onChange={(e) => setNewWatchlistName(e.target.value)}
              placeholder="Watchlist name"
              className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
              onKeyPress={(e) => e.key === 'Enter' && newWatchlistName.trim() && createWatchlist(newWatchlistName.trim())}
            />
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => newWatchlistName.trim() && createWatchlist(newWatchlistName.trim())}
                disabled={!newWatchlistName.trim()}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Create
              </button>
              <button
                onClick={() => {
                  setShowAddWatchlist(false)
                  setNewWatchlistName('')
                }}
                className="flex-1 px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Ticker Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={newTicker}
          onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
          placeholder="Add ticker (e.g., AAPL)"
          className="flex-1 px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
          onKeyPress={(e) => e.key === 'Enter' && newTicker.trim() && addTicker(activeWatchlist, newTicker.trim())}
        />
        <button
          onClick={() => newTicker.trim() && addTicker(activeWatchlist, newTicker.trim())}
          disabled={!newTicker.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Add
        </button>
      </div>

      {/* Watchlist Grid */}
      {currentWatchlist && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {currentWatchlist.tickers.map(ticker => {
            const data = marketData[ticker]
            
            return (
              <div
                key={ticker}
                onClick={() => onTickerSelect && onTickerSelect(ticker)}
                className="p-4 rounded-xl glass border border-white/10 hover:border-white/20 cursor-pointer transition-all duration-200 hover:shadow-lg"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-white">{ticker}</h3>
                    <Star size={14} className="text-yellow-400" />
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      removeTicker(activeWatchlist, ticker)
                    }}
                    className="p-1 rounded text-gray-400 hover:text-red-400 hover:bg-red-900/20 transition-colors"
                  >
                    <X size={14} />
                  </button>
                </div>
                
                {data ? (
                  <div className="space-y-2">
                    <div className="text-xl font-bold text-white">
                      {formatPrice(data.current_price)}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {data.day_change >= 0 ? (
                        <TrendingUp className="text-green-400" size={16} />
                      ) : (
                        <TrendingDown className="text-red-400" size={16} />
                      )}
                      <div className={`text-sm ${data.day_change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatChange(data.day_change, data.day_change_percent).change}
                      </div>
                    </div>
                    
                    <div className={`text-sm ${data.day_change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatChange(data.day_change, data.day_change_percent).changePercent}
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-400 mt-3">
                      <div>
                        <span className="block">Volume</span>
                        <span className="text-white">{data.volume?.toLocaleString() || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="block">Mkt Cap</span>
                        <span className="text-white">
                          {data.market_cap ? `$${(data.market_cap / 1e9).toFixed(1)}B` : 'N/A'}
                        </span>
                      </div>
                    </div>
                    
                    {data.alerts && data.alerts.length > 0 && (
                      <div className="flex items-center gap-1 mt-2 p-2 rounded bg-yellow-900/20 border border-yellow-900/30">
                        <AlertTriangle size={12} className="text-yellow-400" />
                        <span className="text-xs text-yellow-200">
                          {data.alerts.length} alert{data.alerts.length > 1 ? 's' : ''}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="animate-pulse space-y-2">
                    <div className="h-6 bg-gray-700 rounded"></div>
                    <div className="h-4 bg-gray-700 rounded w-2/3"></div>
                    <div className="h-4 bg-gray-700 rounded w-1/2"></div>
                  </div>
                )}
              </div>
            )
          })}
          
          {currentWatchlist.tickers.length === 0 && (
            <div className="col-span-full p-8 text-center text-gray-400 border-2 border-dashed border-gray-700 rounded-xl">
              <Star size={32} className="mx-auto mb-2 opacity-50" />
              <p>No tickers in this watchlist</p>
              <p className="text-sm">Add some tickers to get started</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default WatchlistManager