import React, { useState, useEffect, useMemo } from 'react'
import { Target, TrendingUp, BarChart3, Settings, Play, RefreshCw, Download } from 'lucide-react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter } from 'recharts'

const PortfolioOptimizer = ({ currentPortfolio, availableAssets, onOptimizationComplete }) => {
  const [optimizationSettings, setOptimizationSettings] = useState({
    objective: 'max_sharpe', // max_sharpe, min_risk, max_return
    riskTolerance: 'moderate', // conservative, moderate, aggressive
    constraints: {
      maxWeight: 0.3,
      minWeight: 0.05,
      maxAssets: 10,
      sectors: {
        technology: 0.4,
        healthcare: 0.3,
        finance: 0.3,
        energy: 0.2,
        consumer: 0.3
      }
    },
    rebalanceFrequency: 'quarterly' // monthly, quarterly, annually
  })
  
  const [optimizationResults, setOptimizationResults] = useState(null)
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [activeTab, setActiveTab] = useState('settings')
  const [backtestResults, setBacktestResults] = useState(null)

  // Generate efficient frontier data
  const efficientFrontier = useMemo(() => {
    const data = []
    for (let risk = 0.05; risk <= 0.3; risk += 0.01) {
      const expectedReturn = 0.03 + (risk - 0.05) * 0.4 + Math.random() * 0.02
      data.push({
        risk: risk,
        return: expectedReturn,
        sharpe: expectedReturn / risk
      })
    }
    return data
  }, [])

  // Current portfolio metrics
  const currentMetrics = useMemo(() => {
    if (!currentPortfolio || currentPortfolio.length === 0) return null
    
    const totalValue = currentPortfolio.reduce((sum, position) => sum + position.currentValue, 0)
    const weightedReturn = currentPortfolio.reduce((sum, position) => {
      const weight = position.currentValue / totalValue
      return sum + (weight * (position.expectedReturn || 0.08))
    }, 0)
    
    const portfolioRisk = Math.sqrt(
      currentPortfolio.reduce((sum, position) => {
        const weight = position.currentValue / totalValue
        return sum + Math.pow(weight * (position.volatility || 0.2), 2)
      }, 0)
    )
    
    return {
      expectedReturn: weightedReturn,
      risk: portfolioRisk,
      sharpeRatio: weightedReturn / portfolioRisk,
      totalValue
    }
  }, [currentPortfolio])

  const runOptimization = async () => {
    setIsOptimizing(true)
    try {
      const response = await fetch('http://localhost:8000/api/portfolio/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          current_portfolio: currentPortfolio,
          available_assets: availableAssets,
          settings: optimizationSettings
        })
      })

      if (response.ok) {
        const results = await response.json()
        setOptimizationResults(results)
        
        if (onOptimizationComplete) {
          onOptimizationComplete(results)
        }
      }
    } catch (error) {
      console.error('Optimization error:', error)
    } finally {
      setIsOptimizing(false)
    }
  }

  const runBacktest = async () => {
    if (!optimizationResults) return
    
    try {
      const response = await fetch('http://localhost:8000/api/portfolio/backtest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          portfolio: optimizationResults.optimal_portfolio,
          start_date: '2022-01-01',
          end_date: '2024-01-01'
        })
      })

      if (response.ok) {
        const results = await response.json()
        setBacktestResults(results)
      }
    } catch (error) {
      console.error('Backtest error:', error)
    }
  }

  const formatPercent = (value) => `${(value * 100).toFixed(2)}%`
  const formatCurrency = (value) => `$${value.toLocaleString()}`

  const tabs = [
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'results', label: 'Results', icon: Target },
    { id: 'backtest', label: 'Backtest', icon: BarChart3 },
    { id: 'frontier', label: 'Efficient Frontier', icon: TrendingUp }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-4 rounded-xl glass">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Target size={20} />
            Portfolio Optimizer
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={runOptimization}
              disabled={isOptimizing}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {isOptimizing ? (
                <RefreshCw className="animate-spin" size={16} />
              ) : (
                <Play size={16} />
              )}
              {isOptimizing ? 'Optimizing...' : 'Optimize'}
            </button>
          </div>
        </div>

        {/* Current Portfolio Summary */}
        {currentMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
            <div className="text-center p-3 rounded-lg bg-gray-800/50">
              <div className="text-sm text-gray-400">Portfolio Value</div>
              <div className="text-lg font-bold text-white">
                {formatCurrency(currentMetrics.totalValue)}
              </div>
            </div>
            <div className="text-center p-3 rounded-lg bg-gray-800/50">
              <div className="text-sm text-gray-400">Expected Return</div>
              <div className="text-lg font-bold text-green-400">
                {formatPercent(currentMetrics.expectedReturn)}
              </div>
            </div>
            <div className="text-center p-3 rounded-lg bg-gray-800/50">
              <div className="text-sm text-gray-400">Risk (Volatility)</div>
              <div className="text-lg font-bold text-orange-400">
                {formatPercent(currentMetrics.risk)}
              </div>
            </div>
            <div className="text-center p-3 rounded-lg bg-gray-800/50">
              <div className="text-sm text-gray-400">Sharpe Ratio</div>
              <div className="text-lg font-bold text-white">
                {currentMetrics.sharpeRatio.toFixed(2)}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2">
        {tabs.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="rounded-xl glass overflow-hidden">
        {activeTab === 'settings' && (
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Optimization Objective */}
              <div>
                <h4 className="text-lg font-semibold text-white mb-4">Optimization Objective</h4>
                <div className="space-y-3">
                  {[
                    { value: 'max_sharpe', label: 'Maximize Sharpe Ratio', desc: 'Best risk-adjusted returns' },
                    { value: 'min_risk', label: 'Minimize Risk', desc: 'Lowest portfolio volatility' },
                    { value: 'max_return', label: 'Maximize Return', desc: 'Highest expected returns' }
                  ].map(option => (
                    <label key={option.value} className="flex items-start gap-3 p-3 rounded-lg bg-gray-800/50 cursor-pointer hover:bg-gray-800/70">
                      <input
                        type="radio"
                        name="objective"
                        value={option.value}
                        checked={optimizationSettings.objective === option.value}
                        onChange={(e) => setOptimizationSettings(prev => ({
                          ...prev,
                          objective: e.target.value
                        }))}
                        className="mt-1"
                      />
                      <div>
                        <div className="text-white font-medium">{option.label}</div>
                        <div className="text-sm text-gray-400">{option.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Risk Tolerance */}
              <div>
                <h4 className="text-lg font-semibold text-white mb-4">Risk Tolerance</h4>
                <div className="space-y-3">
                  {[
                    { value: 'conservative', label: 'Conservative', desc: 'Lower risk, stable returns' },
                    { value: 'moderate', label: 'Moderate', desc: 'Balanced risk and return' },
                    { value: 'aggressive', label: 'Aggressive', desc: 'Higher risk, higher potential returns' }
                  ].map(option => (
                    <label key={option.value} className="flex items-start gap-3 p-3 rounded-lg bg-gray-800/50 cursor-pointer hover:bg-gray-800/70">
                      <input
                        type="radio"
                        name="riskTolerance"
                        value={option.value}
                        checked={optimizationSettings.riskTolerance === option.value}
                        onChange={(e) => setOptimizationSettings(prev => ({
                          ...prev,
                          riskTolerance: e.target.value
                        }))}
                        className="mt-1"
                      />
                      <div>
                        <div className="text-white font-medium">{option.label}</div>
                        <div className="text-sm text-gray-400">{option.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Constraints */}
            <div>
              <h4 className="text-lg font-semibold text-white mb-4">Portfolio Constraints</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Maximum Weight per Asset
                  </label>
                  <input
                    type="number"
                    min="0.05"
                    max="1"
                    step="0.05"
                    value={optimizationSettings.constraints.maxWeight}
                    onChange={(e) => setOptimizationSettings(prev => ({
                      ...prev,
                      constraints: {
                        ...prev.constraints,
                        maxWeight: parseFloat(e.target.value)
                      }
                    }))}
                    className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Minimum Weight per Asset
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="0.2"
                    step="0.01"
                    value={optimizationSettings.constraints.minWeight}
                    onChange={(e) => setOptimizationSettings(prev => ({
                      ...prev,
                      constraints: {
                        ...prev.constraints,
                        minWeight: parseFloat(e.target.value)
                      }
                    }))}
                    className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Maximum Number of Assets
                  </label>
                  <input
                    type="number"
                    min="3"
                    max="20"
                    value={optimizationSettings.constraints.maxAssets}
                    onChange={(e) => setOptimizationSettings(prev => ({
                      ...prev,
                      constraints: {
                        ...prev.constraints,
                        maxAssets: parseInt(e.target.value)
                      }
                    }))}
                    className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Rebalance Frequency
                  </label>
                  <select
                    value={optimizationSettings.rebalanceFrequency}
                    onChange={(e) => setOptimizationSettings(prev => ({
                      ...prev,
                      rebalanceFrequency: e.target.value
                    }))}
                    className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="monthly">Monthly</option>
                    <option value="quarterly">Quarterly</option>
                    <option value="annually">Annually</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="p-6">
            {optimizationResults ? (
              <div className="space-y-6">
                {/* Optimization Summary */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Expected Return</div>
                    <div className="text-xl font-bold text-green-400">
                      {formatPercent(optimizationResults.expected_return)}
                    </div>
                    {currentMetrics && (
                      <div className="text-sm text-gray-400">
                        vs {formatPercent(currentMetrics.expectedReturn)} current
                      </div>
                    )}
                  </div>
                  
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Risk (Volatility)</div>
                    <div className="text-xl font-bold text-orange-400">
                      {formatPercent(optimizationResults.risk)}
                    </div>
                    {currentMetrics && (
                      <div className="text-sm text-gray-400">
                        vs {formatPercent(currentMetrics.risk)} current
                      </div>
                    )}
                  </div>
                  
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Sharpe Ratio</div>
                    <div className="text-xl font-bold text-white">
                      {optimizationResults.sharpe_ratio.toFixed(2)}
                    </div>
                    {currentMetrics && (
                      <div className="text-sm text-gray-400">
                        vs {currentMetrics.sharpeRatio.toFixed(2)} current
                      </div>
                    )}
                  </div>
                </div>

                {/* Optimal Portfolio Allocation */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white">Optimal Portfolio Allocation</h4>
                    <button
                      onClick={runBacktest}
                      className="flex items-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
                    >
                      <BarChart3 size={16} />
                      Run Backtest
                    </button>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-gray-800/50">
                        <tr>
                          <th className="text-left p-3 text-sm font-medium text-gray-300">Asset</th>
                          <th className="text-right p-3 text-sm font-medium text-gray-300">Weight</th>
                          <th className="text-right p-3 text-sm font-medium text-gray-300">Current</th>
                          <th className="text-right p-3 text-sm font-medium text-gray-300">Target</th>
                          <th className="text-right p-3 text-sm font-medium text-gray-300">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {optimizationResults.optimal_portfolio.map((asset, index) => {
                          const currentWeight = currentPortfolio?.find(p => p.ticker === asset.ticker)?.weight || 0
                          const change = asset.weight - currentWeight
                          
                          return (
                            <tr key={index} className="border-t border-white/10">
                              <td className="p-3 font-medium text-white">{asset.ticker}</td>
                              <td className="p-3 text-right text-white">{formatPercent(asset.weight)}</td>
                              <td className="p-3 text-right text-gray-400">{formatPercent(currentWeight)}</td>
                              <td className="p-3 text-right text-white font-semibold">
                                {formatCurrency(asset.target_value)}
                              </td>
                              <td className={`p-3 text-right font-semibold ${
                                change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-gray-400'
                              }`}>
                                {change > 0 ? '+' : ''}{formatPercent(change)}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-4">
                  <button className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold">
                    Apply Optimization
                  </button>
                  <button className="px-6 py-3 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors">
                    <Download size={20} />
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center p-8">
                <Target className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-gray-400">Run optimization to see results</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'backtest' && (
          <div className="p-6">
            {backtestResults ? (
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-white">Backtest Results</h4>
                
                {/* Performance Chart */}
                <div style={{ width: '100%', height: 400 }}>
                  <ResponsiveContainer>
                    <LineChart data={backtestResults.performance_data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="date" 
                        stroke="#9ca3af"
                        tickFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="portfolio_value" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        name="Optimized Portfolio"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="benchmark_value" 
                        stroke="#6b7280" 
                        strokeWidth={2}
                        name="Benchmark"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Performance Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Total Return</div>
                    <div className="text-xl font-bold text-green-400">
                      {formatPercent(backtestResults.total_return)}
                    </div>
                  </div>
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Annual Return</div>
                    <div className="text-xl font-bold text-white">
                      {formatPercent(backtestResults.annual_return)}
                    </div>
                  </div>
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Max Drawdown</div>
                    <div className="text-xl font-bold text-red-400">
                      {formatPercent(backtestResults.max_drawdown)}
                    </div>
                  </div>
                  <div className="text-center p-4 rounded-lg bg-gray-800/50">
                    <div className="text-sm text-gray-400">Sharpe Ratio</div>
                    <div className="text-xl font-bold text-white">
                      {backtestResults.sharpe_ratio.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center p-8">
                <BarChart3 className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-gray-400">Run backtest to see historical performance</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'frontier' && (
          <div className="p-6">
            <h4 className="text-lg font-semibold text-white mb-4">Efficient Frontier</h4>
            <div style={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <ScatterChart data={efficientFrontier}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    type="number" 
                    dataKey="risk" 
                    name="Risk"
                    stroke="#9ca3af"
                    tickFormatter={formatPercent}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="return" 
                    name="Return"
                    stroke="#9ca3af"
                    tickFormatter={formatPercent}
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      formatPercent(value),
                      name === 'risk' ? 'Risk' : name === 'return' ? 'Return' : 'Sharpe Ratio'
                    ]}
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  />
                  <Scatter 
                    dataKey="return" 
                    fill="#3b82f6"
                    name="Efficient Portfolios"
                  />
                  {currentMetrics && (
                    <Scatter
                      data={[{ risk: currentMetrics.risk, return: currentMetrics.expectedReturn }]}
                      fill="#ef4444"
                      name="Current Portfolio"
                    />
                  )}
                  {optimizationResults && (
                    <Scatter
                      data={[{ risk: optimizationResults.risk, return: optimizationResults.expected_return }]}
                      fill="#10b981"
                      name="Optimized Portfolio"
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default PortfolioOptimizer