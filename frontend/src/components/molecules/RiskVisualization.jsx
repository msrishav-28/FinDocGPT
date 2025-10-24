import React, { useState, useMemo } from 'react'
import { AlertTriangle, TrendingDown, Shield, BarChart3, PieChart } from 'lucide-react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, PieChart as RechartsPieChart, Cell } from 'recharts'

const RiskVisualization = ({ riskData, portfolioData, ticker }) => {
  const [activeView, setActiveView] = useState('overview')
  const [timeframe, setTimeframe] = useState('1Y')

  // Generate risk metrics data
  const riskMetrics = useMemo(() => {
    if (!riskData) return null

    return {
      volatility: riskData.volatility || 0.25,
      beta: riskData.beta || 1.2,
      sharpeRatio: riskData.sharpe_ratio || 0.8,
      maxDrawdown: riskData.max_drawdown || 0.15,
      var95: riskData.var_95 || 0.05,
      var99: riskData.var_99 || 0.08,
      expectedReturn: riskData.expected_return || 0.12,
      riskScore: riskData.risk_score || 65
    }
  }, [riskData])

  // Generate historical volatility data
  const volatilityData = useMemo(() => {
    const data = []
    const baseVol = riskMetrics?.volatility || 0.25
    
    for (let i = 0; i < 252; i++) { // Trading days in a year
      const date = new Date()
      date.setDate(date.getDate() - (252 - i))
      
      data.push({
        date: date.toISOString().split('T')[0],
        volatility: baseVol + (Math.random() - 0.5) * 0.1,
        rollingVol: baseVol + Math.sin(i / 50) * 0.05
      })
    }
    
    return data
  }, [riskMetrics])

  // Generate risk decomposition data
  const riskDecomposition = useMemo(() => [
    { name: 'Market Risk', value: 40, color: '#ef4444' },
    { name: 'Sector Risk', value: 25, color: '#f97316' },
    { name: 'Company Risk', value: 20, color: '#eab308' },
    { name: 'Currency Risk', value: 10, color: '#22c55e' },
    { name: 'Other', value: 5, color: '#6b7280' }
  ], [])

  // Generate risk-return scatter data
  const riskReturnData = useMemo(() => {
    const data = []
    const sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
    
    sectors.forEach(sector => {
      data.push({
        sector,
        risk: Math.random() * 0.3 + 0.1,
        return: Math.random() * 0.2 + 0.05,
        size: Math.random() * 100 + 50
      })
    })
    
    // Add current stock
    data.push({
      sector: ticker,
      risk: riskMetrics?.volatility || 0.25,
      return: riskMetrics?.expectedReturn || 0.12,
      size: 150,
      isTarget: true
    })
    
    return data
  }, [riskMetrics, ticker])

  // Generate radar chart data for risk factors
  const radarData = useMemo(() => [
    {
      factor: 'Market Risk',
      value: 75,
      fullMark: 100
    },
    {
      factor: 'Credit Risk',
      value: 45,
      fullMark: 100
    },
    {
      factor: 'Liquidity Risk',
      value: 30,
      fullMark: 100
    },
    {
      factor: 'Operational Risk',
      value: 55,
      fullMark: 100
    },
    {
      factor: 'Regulatory Risk',
      value: 65,
      fullMark: 100
    },
    {
      factor: 'ESG Risk',
      value: 40,
      fullMark: 100
    }
  ], [])

  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`
  const formatCurrency = (value) => `$${value.toFixed(2)}`

  const getRiskColor = (score) => {
    if (score < 30) return 'text-green-400'
    if (score < 60) return 'text-yellow-400'
    if (score < 80) return 'text-orange-400'
    return 'text-red-400'
  }

  const views = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'volatility', label: 'Volatility', icon: TrendingDown },
    { id: 'decomposition', label: 'Risk Breakdown', icon: PieChart },
    { id: 'factors', label: 'Risk Factors', icon: AlertTriangle }
  ]

  if (!riskMetrics) {
    return (
      <div className="p-8 text-center rounded-xl glass">
        <Shield className="mx-auto text-gray-400 mb-4" size={48} />
        <p className="text-gray-400">No risk data available</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-4 rounded-xl glass">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Shield size={20} />
            Risk Analysis - {ticker}
          </h3>
          <div className="flex items-center gap-2">
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none text-sm"
            >
              <option value="1M">1 Month</option>
              <option value="3M">3 Months</option>
              <option value="6M">6 Months</option>
              <option value="1Y">1 Year</option>
              <option value="2Y">2 Years</option>
            </select>
          </div>
        </div>

        {/* Risk Score */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 rounded-lg bg-gray-800/50">
            <div className="text-sm text-gray-400 mb-1">Risk Score</div>
            <div className={`text-2xl font-bold ${getRiskColor(riskMetrics.riskScore)}`}>
              {riskMetrics.riskScore}/100
            </div>
          </div>
          
          <div className="text-center p-4 rounded-lg bg-gray-800/50">
            <div className="text-sm text-gray-400 mb-1">Volatility</div>
            <div className="text-2xl font-bold text-white">
              {formatPercent(riskMetrics.volatility)}
            </div>
          </div>
          
          <div className="text-center p-4 rounded-lg bg-gray-800/50">
            <div className="text-sm text-gray-400 mb-1">Max Drawdown</div>
            <div className="text-2xl font-bold text-red-400">
              {formatPercent(riskMetrics.maxDrawdown)}
            </div>
          </div>
          
          <div className="text-center p-4 rounded-lg bg-gray-800/50">
            <div className="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
            <div className="text-2xl font-bold text-white">
              {riskMetrics.sharpeRatio.toFixed(2)}
            </div>
          </div>
        </div>
      </div>

      {/* View Navigation */}
      <div className="flex flex-wrap gap-2">
        {views.map(view => {
          const Icon = view.icon
          return (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeView === view.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <Icon size={16} />
              {view.label}
            </button>
          )
        })}
      </div>

      {/* View Content */}
      <div className="rounded-xl glass overflow-hidden">
        {activeView === 'overview' && (
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Risk Metrics Table */}
              <div>
                <h4 className="text-lg font-semibold text-white mb-4">Risk Metrics</h4>
                <div className="space-y-3">
                  <div className="flex justify-between p-3 rounded-lg bg-gray-800/50">
                    <span className="text-gray-400">Beta</span>
                    <span className="text-white font-semibold">{riskMetrics.beta.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between p-3 rounded-lg bg-gray-800/50">
                    <span className="text-gray-400">Value at Risk (95%)</span>
                    <span className="text-red-400 font-semibold">{formatPercent(riskMetrics.var95)}</span>
                  </div>
                  <div className="flex justify-between p-3 rounded-lg bg-gray-800/50">
                    <span className="text-gray-400">Value at Risk (99%)</span>
                    <span className="text-red-400 font-semibold">{formatPercent(riskMetrics.var99)}</span>
                  </div>
                  <div className="flex justify-between p-3 rounded-lg bg-gray-800/50">
                    <span className="text-gray-400">Expected Return</span>
                    <span className="text-green-400 font-semibold">{formatPercent(riskMetrics.expectedReturn)}</span>
                  </div>
                </div>
              </div>

              {/* Risk-Return Scatter */}
              <div>
                <h4 className="text-lg font-semibold text-white mb-4">Risk-Return Profile</h4>
                <div style={{ width: '100%', height: 300 }}>
                  <ResponsiveContainer>
                    <LineChart data={riskReturnData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="risk" 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={formatPercent}
                        stroke="#9ca3af"
                      />
                      <YAxis 
                        dataKey="return"
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={formatPercent}
                        stroke="#9ca3af"
                      />
                      <Tooltip 
                        formatter={(value, name) => [
                          name === 'risk' ? formatPercent(value) : formatPercent(value),
                          name === 'risk' ? 'Risk' : 'Return'
                        ]}
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      />
                      <Line 
                        dataKey="return" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeView === 'volatility' && (
          <div className="p-6">
            <h4 className="text-lg font-semibold text-white mb-4">Historical Volatility</h4>
            <div style={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <AreaChart data={volatilityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9ca3af"
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis 
                    stroke="#9ca3af"
                    tickFormatter={formatPercent}
                  />
                  <Tooltip 
                    formatter={(value) => [formatPercent(value), 'Volatility']}
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="volatility" 
                    stroke="#ef4444" 
                    fill="rgba(239, 68, 68, 0.2)"
                    strokeWidth={2}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="rollingVol" 
                    stroke="#3b82f6" 
                    fill="rgba(59, 130, 246, 0.1)"
                    strokeWidth={2}
                  />
                  <Legend />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeView === 'decomposition' && (
          <div className="p-6">
            <h4 className="text-lg font-semibold text-white mb-4">Risk Decomposition</h4>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <RechartsPieChart>
                    <Pie
                      data={riskDecomposition}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {riskDecomposition.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </div>
              
              <div className="space-y-3">
                <h5 className="font-semibold text-white">Risk Components</h5>
                {riskDecomposition.map((item, index) => (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-gray-300">{item.name}</span>
                    </div>
                    <span className="text-white font-semibold">{item.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeView === 'factors' && (
          <div className="p-6">
            <h4 className="text-lg font-semibold text-white mb-4">Risk Factor Analysis</h4>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div style={{ width: '100%', height: 400 }}>
                <ResponsiveContainer>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#374151" />
                    <PolarAngleAxis dataKey="factor" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <PolarRadiusAxis 
                      angle={90} 
                      domain={[0, 100]} 
                      tick={{ fill: '#9ca3af', fontSize: 10 }}
                    />
                    <Radar
                      name="Risk Level"
                      dataKey="value"
                      stroke="#ef4444"
                      fill="rgba(239, 68, 68, 0.2)"
                      strokeWidth={2}
                    />
                    <Tooltip 
                      formatter={(value) => [`${value}%`, 'Risk Level']}
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="space-y-4">
                <h5 className="font-semibold text-white">Risk Factor Details</h5>
                {radarData.map((factor, index) => (
                  <div key={index} className="p-4 rounded-lg bg-gray-800/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-300 font-medium">{factor.factor}</span>
                      <span className={`font-semibold ${getRiskColor(factor.value)}`}>
                        {factor.value}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          factor.value < 30 ? 'bg-green-500' :
                          factor.value < 60 ? 'bg-yellow-500' :
                          factor.value < 80 ? 'bg-orange-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${factor.value}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default RiskVisualization