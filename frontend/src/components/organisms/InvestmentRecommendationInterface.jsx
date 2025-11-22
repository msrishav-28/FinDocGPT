import React, { useState, useEffect } from 'react'
import { Target, TrendingUp, Shield, BarChart3, Search, Filter, RefreshCw } from 'lucide-react'
import InvestmentRecommendationDisplay from './InvestmentRecommendationDisplay'
import { RiskVisualization } from '../molecules'
import PortfolioOptimizer from './PortfolioOptimizer'

const InvestmentRecommendationInterface = ({ userId = 'demo-user' }) => {
  const [activeTab, setActiveTab] = useState('recommendations')
  const [selectedTicker, setSelectedTicker] = useState('')
  const [recommendations, setRecommendations] = useState([])
  const [currentRecommendation, setCurrentRecommendation] = useState(null)
  const [riskData, setRiskData] = useState(null)
  const [portfolioData, setPortfolioData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterSignal, setFilterSignal] = useState('all')

  // Load user's portfolio and recommendations
  useEffect(() => {
    loadPortfolioData()
    loadRecommendations()
  }, [])

  const loadPortfolioData = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/portfolio/${userId}`)
      if (response.ok) {
        const data = await response.json()
        setPortfolioData(data)
      }
    } catch (error) {
      console.error('Error loading portfolio:', error)
    }
  }

  const loadRecommendations = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/recommendations/list?user_id=${userId}`)
      if (response.ok) {
        const data = await response.json()
        setRecommendations(data.recommendations || [])
      }
    } catch (error) {
      console.error('Error loading recommendations:', error)
    }
  }

  const generateRecommendation = async (ticker) => {
    if (!ticker.trim()) return

    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/recommendations/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          user_id: userId,
          include_risk_analysis: true
        })
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentRecommendation(data.recommendation)
        setRiskData(data.risk_analysis)
        
        // Add to recommendations list
        setRecommendations(prev => [data.recommendation, ...prev.filter(r => r.ticker !== ticker.toUpperCase())])
      }
    } catch (error) {
      console.error('Error generating recommendation:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleRecommendationSelect = (recommendation) => {
    setCurrentRecommendation(recommendation)
    setSelectedTicker(recommendation.ticker)
    
    // Load risk data for selected recommendation
    loadRiskData(recommendation.ticker)
  }

  const loadRiskData = async (ticker) => {
    try {
      const response = await fetch(`http://localhost:8000/api/risk/analysis/${ticker}`)
      if (response.ok) {
        const data = await response.json()
        setRiskData(data)
      }
    } catch (error) {
      console.error('Error loading risk data:', error)
    }
  }

  const filteredRecommendations = recommendations.filter(rec => {
    const matchesSearch = rec.ticker.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesFilter = filterSignal === 'all' || rec.signal.toLowerCase() === filterSignal.toLowerCase()
    return matchesSearch && matchesFilter
  })

  const getSignalColor = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'STRONG_BUY':
        return 'bg-green-600'
      case 'BUY':
        return 'bg-green-500'
      case 'HOLD':
        return 'bg-yellow-500'
      case 'SELL':
        return 'bg-red-500'
      case 'STRONG_SELL':
        return 'bg-red-600'
      default:
        return 'bg-gray-500'
    }
  }

  const tabs = [
    { id: 'recommendations', label: 'Recommendations', icon: Target, description: 'AI-powered investment recommendations' },
    { id: 'risk', label: 'Risk Analysis', icon: Shield, description: 'Comprehensive risk assessment' },
    { id: 'optimizer', label: 'Portfolio Optimizer', icon: BarChart3, description: 'Optimize portfolio allocation' }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-6 rounded-xl glass">
        <h2 className="text-2xl font-bold text-white mb-2">Investment Intelligence Center</h2>
        <p className="text-gray-400">
          Get AI-powered investment recommendations with comprehensive risk analysis and portfolio optimization
        </p>
      </div>

      {/* Search and Generate */}
      <div className="p-4 rounded-xl glass">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
              <input
                type="text"
                value={selectedTicker}
                onChange={(e) => setSelectedTicker(e.target.value.toUpperCase())}
                placeholder="Enter ticker symbol (e.g., AAPL, MSFT, TSLA)"
                className="w-full pl-10 pr-4 py-3 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                onKeyPress={(e) => e.key === 'Enter' && generateRecommendation(selectedTicker)}
              />
            </div>
          </div>
          <button
            onClick={() => generateRecommendation(selectedTicker)}
            disabled={!selectedTicker.trim() || loading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold flex items-center gap-2"
          >
            {loading ? (
              <RefreshCw className="animate-spin" size={16} />
            ) : (
              <Target size={16} />
            )}
            {loading ? 'Analyzing...' : 'Get Recommendation'}
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2">
        {tabs.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 rounded-lg transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <Icon size={18} />
              <div className="text-left">
                <div className="font-medium">{tab.label}</div>
                <div className="text-xs opacity-75">{tab.description}</div>
              </div>
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="min-h-[600px]">
        {activeTab === 'recommendations' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Recommendations List */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Recent Recommendations</h3>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search..."
                    className="px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none text-sm"
                  />
                  <select
                    value={filterSignal}
                    onChange={(e) => setFilterSignal(e.target.value)}
                    className="px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none text-sm"
                  >
                    <option value="all">All Signals</option>
                    <option value="strong_buy">Strong Buy</option>
                    <option value="buy">Buy</option>
                    <option value="hold">Hold</option>
                    <option value="sell">Sell</option>
                    <option value="strong_sell">Strong Sell</option>
                  </select>
                </div>
              </div>

              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {filteredRecommendations.length > 0 ? (
                  filteredRecommendations.map((rec, index) => (
                    <div
                      key={index}
                      onClick={() => handleRecommendationSelect(rec)}
                      className={`p-4 rounded-xl cursor-pointer transition-all ${
                        currentRecommendation?.ticker === rec.ticker
                          ? 'glass border-2 border-blue-500'
                          : 'glass border border-white/10 hover:border-white/20'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold text-white">{rec.ticker}</h4>
                        <div className={`px-2 py-1 rounded text-xs text-white ${getSignalColor(rec.signal)}`}>
                          {rec.signal}
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="text-white">{Math.round(rec.confidence * 100)}%</span>
                      </div>
                      
                      {rec.target_price && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Target:</span>
                          <span className="text-green-400">${rec.target_price.toFixed(2)}</span>
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Risk:</span>
                        <span className={`${
                          rec.risk_level === 'low' ? 'text-green-400' :
                          rec.risk_level === 'moderate' ? 'text-yellow-400' :
                          'text-red-400'
                        }`}>
                          {rec.risk_level?.toUpperCase()}
                        </span>
                      </div>
                      
                      <div className="text-xs text-gray-500 mt-2">
                        {new Date(rec.created_at || Date.now()).toLocaleDateString()}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center p-8 rounded-xl glass">
                    <Target className="mx-auto text-gray-400 mb-4" size={48} />
                    <p className="text-gray-400">No recommendations yet</p>
                    <p className="text-sm text-gray-500 mt-2">
                      Enter a ticker symbol to generate your first recommendation
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Recommendation Details */}
            <div className="lg:col-span-2">
              {currentRecommendation ? (
                <InvestmentRecommendationDisplay
                  ticker={currentRecommendation.ticker}
                  recommendation={currentRecommendation}
                  onPositionSizeChange={(size) => {
                    console.log('Position size changed:', size)
                  }}
                />
              ) : (
                <div className="text-center p-8 rounded-xl glass">
                  <TrendingUp className="mx-auto text-gray-400 mb-4" size={48} />
                  <p className="text-gray-400 mb-4">Select a recommendation to view details</p>
                  <p className="text-sm text-gray-500">
                    Or generate a new recommendation using the search box above
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'risk' && (
          <RiskVisualization
            riskData={riskData}
            portfolioData={portfolioData}
            ticker={currentRecommendation?.ticker || selectedTicker}
          />
        )}

        {activeTab === 'optimizer' && (
          <PortfolioOptimizer
            currentPortfolio={portfolioData?.positions || []}
            availableAssets={recommendations.map(r => ({
              ticker: r.ticker,
              expected_return: r.expected_return || 0.08,
              volatility: r.volatility || 0.2,
              recommendation: r.signal
            }))}
            onOptimizationComplete={(results) => {
              console.log('Optimization complete:', results)
            }}
          />
        )}
      </div>
    </div>
  )
}

export default InvestmentRecommendationInterface