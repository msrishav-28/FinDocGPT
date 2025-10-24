import React, { useState, useEffect, useMemo } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, Percent, Target, AlertCircle, Plus, Edit2 } from 'lucide-react'
import AdvancedChart from './AdvancedChart'

const PortfolioTracker = ({ userId = 'demo-user' }) => {
  const [portfolio, setPortfolio] = useState({
    id: 'main-portfolio',
    name: 'Main Portfolio',
    cash: 10000,
    positions: [
      { ticker: 'AAPL', shares: 50, avgCost: 150.00, currentPrice: 175.50 },
      { ticker: 'MSFT', shares: 30, avgCost: 280.00, currentPrice: 310.25 },
      { ticker: 'GOOGL', shares: 10, avgCost: 2500.00, currentPrice: 2650.00 },
      { ticker: 'TSLA', shares: 25, avgCost: 200.00, currentPrice: 185.75 }
    ]
  })
  
  const [marketData, setMarketData] = useState({})
  const [showAddPosition, setShowAddPosition] = useState(false)
  const [newPosition, setNewPosition] = useState({ ticker: '', shares: '', avgCost: '' })
  const [selectedTimeframe, setSelectedTimeframe] = useState('1M')
  const [portfolioHistory, setPortfolioHistory] = useState([])

  // Fetch current market data for portfolio positions
  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        const tickers = portfolio.positions.map(p => p.ticker)
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
        
        setMarketData(dataMap)
      } catch (error) {
        console.error('Error fetching market data:', error)
      }
    }

    if (portfolio.positions.length > 0) {
      fetchMarketData()
    }
  }, [portfolio.positions])

  // Calculate portfolio metrics
  const portfolioMetrics = useMemo(() => {
    let totalValue = portfolio.cash
    let totalCost = portfolio.cash
    let totalGainLoss = 0
    let dayChange = 0

    const positionDetails = portfolio.positions.map(position => {
      const marketPrice = marketData[position.ticker]?.current_price || position.currentPrice
      const dayChangeAmount = marketData[position.ticker]?.day_change || 0
      
      const currentValue = position.shares * marketPrice
      const costBasis = position.shares * position.avgCost
      const gainLoss = currentValue - costBasis
      const gainLossPercent = (gainLoss / costBasis) * 100
      const positionDayChange = position.shares * dayChangeAmount

      totalValue += currentValue
      totalCost += costBasis
      totalGainLoss += gainLoss
      dayChange += positionDayChange

      return {
        ...position,
        currentPrice: marketPrice,
        currentValue,
        costBasis,
        gainLoss,
        gainLossPercent,
        dayChange: positionDayChange,
        weight: 0 // Will be calculated after totalValue is known
      }
    })

    // Calculate position weights
    positionDetails.forEach(position => {
      position.weight = (position.currentValue / totalValue) * 100
    })

    const totalGainLossPercent = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0
    const dayChangePercent = totalValue > 0 ? (dayChange / (totalValue - dayChange)) * 100 : 0

    return {
      totalValue,
      totalCost,
      totalGainLoss,
      totalGainLossPercent,
      dayChange,
      dayChangePercent,
      positions: positionDetails,
      cashWeight: (portfolio.cash / totalValue) * 100
    }
  }, [portfolio, marketData])

  const addPosition = () => {
    if (!newPosition.ticker || !newPosition.shares || !newPosition.avgCost) return

    const position = {
      ticker: newPosition.ticker.toUpperCase(),
      shares: parseFloat(newPosition.shares),
      avgCost: parseFloat(newPosition.avgCost),
      currentPrice: parseFloat(newPosition.avgCost) // Will be updated by market data
    }

    setPortfolio(prev => ({
      ...prev,
      positions: [...prev.positions, position]
    }))

    setNewPosition({ ticker: '', shares: '', avgCost: '' })
    setShowAddPosition(false)
  }

  const removePosition = (ticker) => {
    setPortfolio(prev => ({
      ...prev,
      positions: prev.positions.filter(p => p.ticker !== ticker)
    }))
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount)
  }

  const formatPercent = (percent) => {
    const sign = percent >= 0 ? '+' : ''
    return `${sign}${percent.toFixed(2)}%`
  }

  // Prepare data for pie chart
  const pieChartData = [
    ...portfolioMetrics.positions.map(position => ({
      name: position.ticker,
      value: position.currentValue,
      color: `hsl(${Math.random() * 360}, 70%, 50%)`
    })),
    ...(portfolioMetrics.cashWeight > 1 ? [{
      name: 'Cash',
      value: portfolio.cash,
      color: '#6b7280'
    }] : [])
  ]

  // Prepare data for performance chart
  const performanceData = portfolioMetrics.positions.map(position => ({
    ticker: position.ticker,
    gainLoss: position.gainLossPercent,
    dayChange: (position.dayChange / position.currentValue) * 100
  }))

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl glass">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="text-blue-400" size={20} />
            <span className="text-sm text-gray-400">Total Value</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {formatCurrency(portfolioMetrics.totalValue)}
          </div>
          <div className={`text-sm flex items-center gap-1 ${
            portfolioMetrics.dayChange >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {portfolioMetrics.dayChange >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
            {formatCurrency(portfolioMetrics.dayChange)} ({formatPercent(portfolioMetrics.dayChangePercent)})
          </div>
        </div>

        <div className="p-4 rounded-xl glass">
          <div className="flex items-center gap-2 mb-2">
            <Target className="text-green-400" size={20} />
            <span className="text-sm text-gray-400">Total Gain/Loss</span>
          </div>
          <div className={`text-2xl font-bold ${
            portfolioMetrics.totalGainLoss >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {formatCurrency(portfolioMetrics.totalGainLoss)}
          </div>
          <div className={`text-sm ${
            portfolioMetrics.totalGainLoss >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {formatPercent(portfolioMetrics.totalGainLossPercent)}
          </div>
        </div>

        <div className="p-4 rounded-xl glass">
          <div className="flex items-center gap-2 mb-2">
            <Percent className="text-purple-400" size={20} />
            <span className="text-sm text-gray-400">Positions</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {portfolio.positions.length}
          </div>
          <div className="text-sm text-gray-400">
            {formatPercent(100 - portfolioMetrics.cashWeight)} invested
          </div>
        </div>

        <div className="p-4 rounded-xl glass">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="text-gray-400" size={20} />
            <span className="text-sm text-gray-400">Cash</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {formatCurrency(portfolio.cash)}
          </div>
          <div className="text-sm text-gray-400">
            {formatPercent(portfolioMetrics.cashWeight)} of portfolio
          </div>
        </div>
      </div>

      {/* Portfolio Allocation and Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Allocation Pie Chart */}
        <div className="p-4 rounded-xl glass">
          <h3 className="text-lg font-semibold text-white mb-4">Portfolio Allocation</h3>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={pieChartData}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                >
                  {pieChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Performance Bar Chart */}
        <div className="p-4 rounded-xl glass">
          <h3 className="text-lg font-semibold text-white mb-4">Position Performance</h3>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="ticker" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  formatter={(value) => `${value.toFixed(2)}%`}
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                />
                <Legend />
                <Bar dataKey="gainLoss" fill="#10b981" name="Total Gain/Loss %" />
                <Bar dataKey="dayChange" fill="#3b82f6" name="Day Change %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="rounded-xl glass overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <h3 className="text-lg font-semibold text-white">Positions</h3>
          <button
            onClick={() => setShowAddPosition(true)}
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus size={16} />
            Add Position
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="text-left p-4 text-sm font-medium text-gray-300">Symbol</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Shares</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Avg Cost</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Current Price</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Market Value</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Gain/Loss</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Day Change</th>
                <th className="text-right p-4 text-sm font-medium text-gray-300">Weight</th>
                <th className="text-center p-4 text-sm font-medium text-gray-300">Actions</th>
              </tr>
            </thead>
            <tbody>
              {portfolioMetrics.positions.map((position) => (
                <tr key={position.ticker} className="border-t border-white/10 hover:bg-white/5">
                  <td className="p-4">
                    <div className="font-semibold text-white">{position.ticker}</div>
                  </td>
                  <td className="p-4 text-right text-gray-300">{position.shares}</td>
                  <td className="p-4 text-right text-gray-300">{formatCurrency(position.avgCost)}</td>
                  <td className="p-4 text-right text-white">{formatCurrency(position.currentPrice)}</td>
                  <td className="p-4 text-right text-white font-semibold">{formatCurrency(position.currentValue)}</td>
                  <td className={`p-4 text-right font-semibold ${
                    position.gainLoss >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(position.gainLoss)}
                    <div className="text-sm">({formatPercent(position.gainLossPercent)})</div>
                  </td>
                  <td className={`p-4 text-right ${
                    position.dayChange >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(position.dayChange)}
                  </td>
                  <td className="p-4 text-right text-gray-300">{position.weight.toFixed(1)}%</td>
                  <td className="p-4 text-center">
                    <button
                      onClick={() => removePosition(position.ticker)}
                      className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                      title="Remove position"
                    >
                      <AlertCircle size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Position Modal */}
      {showAddPosition && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Add New Position</h3>
            <div className="space-y-4">
              <input
                type="text"
                value={newPosition.ticker}
                onChange={(e) => setNewPosition(prev => ({ ...prev, ticker: e.target.value.toUpperCase() }))}
                placeholder="Ticker (e.g., AAPL)"
                className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
              <input
                type="number"
                value={newPosition.shares}
                onChange={(e) => setNewPosition(prev => ({ ...prev, shares: e.target.value }))}
                placeholder="Number of shares"
                className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
              <input
                type="number"
                step="0.01"
                value={newPosition.avgCost}
                onChange={(e) => setNewPosition(prev => ({ ...prev, avgCost: e.target.value }))}
                placeholder="Average cost per share"
                className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div className="flex gap-2 mt-6">
              <button
                onClick={addPosition}
                disabled={!newPosition.ticker || !newPosition.shares || !newPosition.avgCost}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Add Position
              </button>
              <button
                onClick={() => {
                  setShowAddPosition(false)
                  setNewPosition({ ticker: '', shares: '', avgCost: '' })
                }}
                className="flex-1 px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default PortfolioTracker