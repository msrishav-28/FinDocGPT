import React, { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, Target, AlertTriangle, Info, Star, DollarSign, Percent, BarChart3 } from 'lucide-react'
import AdvancedChart from './AdvancedChart'

const InvestmentRecommendationDisplay = ({ ticker, recommendation, onPositionSizeChange }) => {
  const [expandedSection, setExpandedSection] = useState('overview')
  const [positionSize, setPositionSize] = useState(recommendation?.suggested_position_size || 0)
  const [riskTolerance, setRiskTolerance] = useState('moderate')

  useEffect(() => {
    if (recommendation?.suggested_position_size) {
      setPositionSize(recommendation.suggested_position_size)
    }
  }, [recommendation])

  if (!recommendation) {
    return (
      <div className="p-8 text-center rounded-xl glass">
        <Target className="mx-auto text-gray-400 mb-4" size={48} />
        <p className="text-gray-400">No recommendation available</p>
        <p className="text-sm text-gray-500 mt-2">
          Generate a recommendation to see detailed analysis
        </p>
      </div>
    )
  }

  const getSignalColor = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'STRONG_BUY':
        return 'text-green-400 bg-green-900/20 border-green-900/30'
      case 'BUY':
        return 'text-green-300 bg-green-900/15 border-green-900/25'
      case 'HOLD':
        return 'text-yellow-400 bg-yellow-900/20 border-yellow-900/30'
      case 'SELL':
        return 'text-red-300 bg-red-900/15 border-red-900/25'
      case 'STRONG_SELL':
        return 'text-red-400 bg-red-900/20 border-red-900/30'
      default:
        return 'text-gray-400 bg-gray-900/20 border-gray-900/30'
    }
  }

  const getSignalIcon = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'STRONG_BUY':
      case 'BUY':
        return <TrendingUp size={20} />
      case 'SELL':
      case 'STRONG_SELL':
        return <TrendingDown size={20} />
      case 'HOLD':
      default:
        return <Target size={20} />
    }
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return 'text-green-400'
      case 'moderate':
        return 'text-yellow-400'
      case 'high':
        return 'text-orange-400'
      case 'very_high':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
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
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`
  }

  const calculatePositionMetrics = () => {
    const currentPrice = recommendation.current_price || 0
    const targetPrice = recommendation.target_price || currentPrice
    const stopLoss = recommendation.stop_loss || currentPrice * 0.9
    
    const shares = Math.floor(positionSize / currentPrice)
    const totalInvestment = shares * currentPrice
    const potentialGain = shares * (targetPrice - currentPrice)
    const potentialLoss = shares * (currentPrice - stopLoss)
    const riskRewardRatio = potentialLoss > 0 ? potentialGain / potentialLoss : 0

    return {
      shares,
      totalInvestment,
      potentialGain,
      potentialLoss,
      riskRewardRatio,
      gainPercent: currentPrice > 0 ? ((targetPrice - currentPrice) / currentPrice) * 100 : 0,
      lossPercent: currentPrice > 0 ? ((currentPrice - stopLoss) / currentPrice) * 100 : 0
    }
  }

  const positionMetrics = calculatePositionMetrics()

  const sections = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'analysis', label: 'Analysis', icon: BarChart3 },
    { id: 'risk', label: 'Risk Assessment', icon: AlertTriangle },
    { id: 'position', label: 'Position Sizing', icon: DollarSign }
  ]

  return (
    <div className="space-y-6">
      {/* Recommendation Header */}
      <div className="p-6 rounded-xl glass">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white">{ticker}</h2>
            <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${getSignalColor(recommendation.signal)}`}>
              {getSignalIcon(recommendation.signal)}
              <span className="font-semibold">{recommendation.signal}</span>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-gray-400">Confidence</div>
            <div className="flex items-center gap-2">
              <div className="w-20 bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    recommendation.confidence > 0.8 ? 'bg-green-500' :
                    recommendation.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${recommendation.confidence * 100}%` }}
                />
              </div>
              <span className="text-white font-semibold">
                {Math.round(recommendation.confidence * 100)}%
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-sm text-gray-400">Current Price</div>
            <div className="text-xl font-bold text-white">
              {formatCurrency(recommendation.current_price)}
            </div>
          </div>
          
          {recommendation.target_price && (
            <div className="text-center">
              <div className="text-sm text-gray-400">Target Price</div>
              <div className="text-xl font-bold text-green-400">
                {formatCurrency(recommendation.target_price)}
              </div>
              <div className="text-sm text-green-300">
                {formatPercent(((recommendation.target_price - recommendation.current_price) / recommendation.current_price) * 100)}
              </div>
            </div>
          )}
          
          <div className="text-center">
            <div className="text-sm text-gray-400">Risk Level</div>
            <div className={`text-xl font-bold ${getRiskColor(recommendation.risk_level)}`}>
              {recommendation.risk_level?.toUpperCase()}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-sm text-gray-400">Time Horizon</div>
            <div className="text-xl font-bold text-white">
              {recommendation.time_horizon || '3-6M'}
            </div>
          </div>
        </div>
      </div>

      {/* Section Navigation */}
      <div className="flex flex-wrap gap-2">
        {sections.map(section => {
          const Icon = section.icon
          return (
            <button
              key={section.id}
              onClick={() => setExpandedSection(section.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                expandedSection === section.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <Icon size={16} />
              {section.label}
            </button>
          )
        })}
      </div>

      {/* Section Content */}
      <div className="rounded-xl glass overflow-hidden">
        {expandedSection === 'overview' && (
          <div className="p-6 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-3">Investment Thesis</h3>
              <p className="text-gray-300 leading-relaxed">
                {recommendation.reasoning || 'No detailed reasoning provided.'}
              </p>
            </div>

            {recommendation.supporting_factors && recommendation.supporting_factors.length > 0 && (
              <div>
                <h4 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
                  <TrendingUp className="text-green-400" size={16} />
                  Supporting Factors
                </h4>
                <ul className="space-y-2">
                  {recommendation.supporting_factors.map((factor, index) => (
                    <li key={index} className="flex items-start gap-2 text-gray-300">
                      <div className="w-2 h-2 rounded-full bg-green-400 mt-2 flex-shrink-0" />
                      {factor}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {recommendation.risk_factors && recommendation.risk_factors.length > 0 && (
              <div>
                <h4 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
                  <AlertTriangle className="text-red-400" size={16} />
                  Risk Factors
                </h4>
                <ul className="space-y-2">
                  {recommendation.risk_factors.map((factor, index) => (
                    <li key={index} className="flex items-start gap-2 text-gray-300">
                      <div className="w-2 h-2 rounded-full bg-red-400 mt-2 flex-shrink-0" />
                      {factor}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {expandedSection === 'analysis' && (
          <div className="p-6 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Technical & Fundamental Analysis</h3>
              
              {recommendation.analysis_details && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {recommendation.analysis_details.technical && (
                    <div className="p-4 rounded-lg bg-gray-800/50">
                      <h4 className="font-semibold text-white mb-3">Technical Indicators</h4>
                      <div className="space-y-2 text-sm">
                        {Object.entries(recommendation.analysis_details.technical).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>
                            <span className="text-white">{value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {recommendation.analysis_details.fundamental && (
                    <div className="p-4 rounded-lg bg-gray-800/50">
                      <h4 className="font-semibold text-white mb-3">Fundamental Metrics</h4>
                      <div className="space-y-2 text-sm">
                        {Object.entries(recommendation.analysis_details.fundamental).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>
                            <span className="text-white">{value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {recommendation.model_scores && (
              <div>
                <h4 className="text-md font-semibold text-white mb-3">Model Scores</h4>
                <div className="space-y-3">
                  {Object.entries(recommendation.model_scores).map(([model, score]) => (
                    <div key={model} className="flex items-center justify-between">
                      <span className="text-gray-300 capitalize">{model.replace('_', ' ')}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-32 bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              score > 0.7 ? 'bg-green-500' :
                              score > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                        <span className="text-white text-sm w-12 text-right">
                          {Math.round(score * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {expandedSection === 'risk' && (
          <div className="p-6 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Risk Assessment</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 rounded-lg bg-gray-800/50 text-center">
                  <div className="text-sm text-gray-400 mb-1">Overall Risk</div>
                  <div className={`text-xl font-bold ${getRiskColor(recommendation.risk_level)}`}>
                    {recommendation.risk_level?.toUpperCase()}
                  </div>
                </div>
                
                <div className="p-4 rounded-lg bg-gray-800/50 text-center">
                  <div className="text-sm text-gray-400 mb-1">Volatility</div>
                  <div className="text-xl font-bold text-white">
                    {recommendation.volatility ? `${recommendation.volatility.toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
                
                <div className="p-4 rounded-lg bg-gray-800/50 text-center">
                  <div className="text-sm text-gray-400 mb-1">Max Drawdown</div>
                  <div className="text-xl font-bold text-red-400">
                    {recommendation.max_drawdown ? `${recommendation.max_drawdown.toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
              </div>

              {recommendation.risk_metrics && (
                <div className="space-y-4">
                  <h4 className="font-semibold text-white">Risk Metrics</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(recommendation.risk_metrics).map(([metric, value]) => (
                      <div key={metric} className="flex justify-between p-3 rounded-lg bg-gray-800/30">
                        <span className="text-gray-400 capitalize">{metric.replace('_', ' ')}</span>
                        <span className="text-white font-semibold">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {expandedSection === 'position' && (
          <div className="p-6 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Position Sizing Calculator</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Position Size ($)
                    </label>
                    <input
                      type="number"
                      value={positionSize}
                      onChange={(e) => {
                        const newSize = parseFloat(e.target.value) || 0
                        setPositionSize(newSize)
                        if (onPositionSizeChange) {
                          onPositionSizeChange(newSize)
                        }
                      }}
                      className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                      placeholder="10000"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Risk Tolerance
                    </label>
                    <select
                      value={riskTolerance}
                      onChange={(e) => setRiskTolerance(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="conservative">Conservative</option>
                      <option value="moderate">Moderate</option>
                      <option value="aggressive">Aggressive</option>
                    </select>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="p-4 rounded-lg bg-gray-800/50">
                    <h4 className="font-semibold text-white mb-3">Position Details</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Shares:</span>
                        <span className="text-white">{positionMetrics.shares.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Total Investment:</span>
                        <span className="text-white">{formatCurrency(positionMetrics.totalInvestment)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Potential Gain:</span>
                        <span className="text-green-400">
                          {formatCurrency(positionMetrics.potentialGain)} ({formatPercent(positionMetrics.gainPercent)})
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Potential Loss:</span>
                        <span className="text-red-400">
                          {formatCurrency(positionMetrics.potentialLoss)} ({formatPercent(-positionMetrics.lossPercent)})
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Risk/Reward Ratio:</span>
                        <span className="text-white">{positionMetrics.riskRewardRatio.toFixed(2)}:1</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {recommendation.stop_loss && recommendation.target_price && (
              <div>
                <h4 className="font-semibold text-white mb-3">Price Targets</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-red-900/20 border border-red-900/30">
                    <span className="text-red-300">Stop Loss</span>
                    <span className="text-white font-semibold">{formatCurrency(recommendation.stop_loss)}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50">
                    <span className="text-gray-300">Current Price</span>
                    <span className="text-white font-semibold">{formatCurrency(recommendation.current_price)}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-green-900/20 border border-green-900/30">
                    <span className="text-green-300">Target Price</span>
                    <span className="text-white font-semibold">{formatCurrency(recommendation.target_price)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold">
          Add to Portfolio
        </button>
        <button className="flex-1 px-6 py-3 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors font-semibold">
          Set Alert
        </button>
        <button className="px-6 py-3 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors">
          <Star size={20} />
        </button>
      </div>
    </div>
  )
}

export default InvestmentRecommendationDisplay