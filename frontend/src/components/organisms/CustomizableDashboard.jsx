import React, { useState, useCallback } from 'react'
import { Responsive, WidthProvider } from 'react-grid-layout'
import { Settings, Plus, X, Move, Maximize2, Minimize2, BarChart3, TrendingUp, Briefcase, Star } from 'lucide-react'
import RealTimeMarketData from './RealTimeMarketData'
import AdvancedChart from './AdvancedChart'
import { SentimentCard, WatchlistManager } from '../molecules'
import { StatCard } from '../atoms'
import PortfolioTracker from './PortfolioTracker'

const ResponsiveGridLayout = WidthProvider(Responsive)

const CustomizableDashboard = () => {
  const [layouts, setLayouts] = useState({
    lg: [
      { i: 'watchlist', x: 0, y: 0, w: 6, h: 8, minW: 4, minH: 6 },
      { i: 'portfolio', x: 6, y: 0, w: 6, h: 8, minW: 4, minH: 6 },
      { i: 'market-data', x: 0, y: 8, w: 12, h: 8, minW: 6, minH: 6 },
      { i: 'sentiment-1', x: 0, y: 16, w: 4, h: 4, minW: 3, minH: 3 },
      { i: 'sentiment-2', x: 4, y: 16, w: 4, h: 4, minW: 3, minH: 3 },
      { i: 'sentiment-3', x: 8, y: 16, w: 4, h: 4, minW: 3, minH: 3 },
      { i: 'chart-1', x: 0, y: 20, w: 8, h: 6, minW: 6, minH: 4 },
      { i: 'stats-1', x: 8, y: 20, w: 4, h: 6, minW: 3, minH: 3 }
    ]
  })
  
  const [widgets, setWidgets] = useState([
    {
      id: 'watchlist',
      title: 'Watchlist Manager',
      type: 'watchlist',
      config: { userId: 'demo-user' }
    },
    {
      id: 'portfolio',
      title: 'Portfolio Tracker',
      type: 'portfolio',
      config: { userId: 'demo-user' }
    },
    {
      id: 'market-data',
      title: 'Real-Time Market Data',
      type: 'market-data',
      config: { watchlist: ['AAPL', 'MSFT', 'GOOGL', 'TSLA'] }
    },
    {
      id: 'sentiment-1',
      title: 'AAPL Sentiment',
      type: 'sentiment',
      config: { ticker: 'AAPL', docId: 'demo_doc' }
    },
    {
      id: 'sentiment-2',
      title: 'MSFT Sentiment',
      type: 'sentiment',
      config: { ticker: 'MSFT', docId: 'demo_doc' }
    },
    {
      id: 'sentiment-3',
      title: 'GOOGL Sentiment',
      type: 'sentiment',
      config: { ticker: 'GOOGL', docId: 'demo_doc' }
    },
    {
      id: 'chart-1',
      title: 'Price Chart',
      type: 'chart',
      config: { 
        ticker: 'AAPL',
        chartType: 'line',
        period: '1d',
        interval: '5m'
      }
    },
    {
      id: 'stats-1',
      title: 'Market Stats',
      type: 'stats',
      config: { 
        stats: [
          { label: 'Active Positions', value: '12', tone: 'brand' },
          { label: 'Total P&L', value: '+$2,450', tone: 'green' },
          { label: 'Win Rate', value: '68%', tone: 'blue' }
        ]
      }
    }
  ])
  
  const [isEditMode, setIsEditMode] = useState(false)
  const [showAddWidget, setShowAddWidget] = useState(false)

  const onLayoutChange = useCallback((layout, layouts) => {
    setLayouts(layouts)
  }, [])

  const removeWidget = useCallback((widgetId) => {
    setWidgets(prev => prev.filter(w => w.id !== widgetId))
    setLayouts(prev => ({
      ...prev,
      lg: prev.lg.filter(l => l.i !== widgetId)
    }))
  }, [])

  const addWidget = useCallback((widgetType, config) => {
    const newId = `${widgetType}-${Date.now()}`
    const newWidget = {
      id: newId,
      title: config.title || `New ${widgetType}`,
      type: widgetType,
      config
    }
    
    setWidgets(prev => [...prev, newWidget])
    
    // Add to layout
    const newLayoutItem = {
      i: newId,
      x: 0,
      y: 0,
      w: 6,
      h: 4,
      minW: 3,
      minH: 3
    }
    
    setLayouts(prev => ({
      ...prev,
      lg: [...prev.lg, newLayoutItem]
    }))
    
    setShowAddWidget(false)
  }, [])

  const renderWidget = (widget) => {
    const { type, config } = widget
    
    switch (type) {
      case 'watchlist':
        return <WatchlistManager {...config} onTickerSelect={(ticker) => console.log('Selected ticker:', ticker)} />
      
      case 'portfolio':
        return <PortfolioTracker {...config} />
      
      case 'market-data':
        return <RealTimeMarketData {...config} />
      
      case 'sentiment':
        return <SentimentCard ticker={config.ticker} docId={config.docId} />
      
      case 'chart':
        return (
          <ChartWidget 
            ticker={config.ticker}
            chartType={config.chartType}
            period={config.period}
            interval={config.interval}
          />
        )
      
      case 'stats':
        return (
          <div className="grid grid-cols-1 gap-3">
            {config.stats.map((stat, index) => (
              <StatCard
                key={index}
                label={stat.label}
                value={stat.value}
                tone={stat.tone}
              />
            ))}
          </div>
        )
      
      default:
        return <div className="p-4 text-gray-400">Unknown widget type: {type}</div>
    }
  }

  return (
    <div className="space-y-4">
      {/* Dashboard Controls */}
      <div className="flex items-center justify-between p-4 rounded-xl glass">
        <h1 className="text-xl font-semibold text-white">Financial Intelligence Dashboard</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAddWidget(true)}
            className="flex items-center gap-2 px-3 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
          >
            <Plus size={16} />
            Add Widget
          </button>
          <button
            onClick={() => setIsEditMode(!isEditMode)}
            className={`flex items-center gap-2 px-3 py-2 rounded-md transition-colors ${
              isEditMode 
                ? 'bg-orange-600 text-white hover:bg-orange-700' 
                : 'bg-gray-600 text-white hover:bg-gray-700'
            }`}
          >
            <Settings size={16} />
            {isEditMode ? 'Exit Edit' : 'Edit Layout'}
          </button>
        </div>
      </div>

      {/* Grid Layout */}
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        onLayoutChange={onLayoutChange}
        breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
        rowHeight={60}
        isDraggable={isEditMode}
        isResizable={isEditMode}
        margin={[16, 16]}
        containerPadding={[0, 0]}
      >
        {widgets.map(widget => (
          <div key={widget.id} className="relative">
            <div className="h-full rounded-xl glass overflow-hidden">
              {/* Widget Header */}
              <div className={`flex items-center justify-between p-3 border-b border-white/10 ${
                isEditMode ? 'bg-gray-800/50' : ''
              }`}>
                <h3 className="text-sm font-medium text-white flex items-center gap-2">
                  {isEditMode && <Move size={14} className="text-gray-400" />}
                  {widget.title}
                </h3>
                {isEditMode && (
                  <button
                    onClick={() => removeWidget(widget.id)}
                    className="p-1 rounded text-gray-400 hover:text-red-400 hover:bg-red-900/20 transition-colors"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
              
              {/* Widget Content */}
              <div className="p-3 h-full overflow-auto">
                {renderWidget(widget)}
              </div>
            </div>
          </div>
        ))}
      </ResponsiveGridLayout>

      {/* Add Widget Modal */}
      {showAddWidget && (
        <AddWidgetModal
          onAdd={addWidget}
          onClose={() => setShowAddWidget(false)}
        />
      )}
    </div>
  )
}

// Chart Widget Component
const ChartWidget = ({ ticker, chartType, period, interval }) => {
  const [chartData, setChartData] = useState(null)
  
  React.useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/market/historical/${ticker}?period=${period}&interval=${interval}`)
        if (response.ok) {
          const data = await response.json()
          
          const processedData = {
            labels: data.data.map(point => new Date(point.timestamp)),
            datasets: [
              {
                label: `${ticker} Price`,
                data: data.data.map(point => point.close),
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: chartType === 'line',
                tension: 0.1
              }
            ]
          }
          
          setChartData(processedData)
        }
      } catch (error) {
        console.error('Error fetching chart data:', error)
      }
    }
    
    fetchData()
  }, [ticker, period, interval, chartType])
  
  return (
    <AdvancedChart
      data={chartData}
      type={chartType}
      height={300}
      showControls={false}
    />
  )
}

// Add Widget Modal Component
const AddWidgetModal = ({ onAdd, onClose }) => {
  const [selectedType, setSelectedType] = useState('chart')
  const [config, setConfig] = useState({})

  const widgetTypes = [
    { id: 'watchlist', name: 'Watchlist Manager', description: 'Customizable watchlists with real-time data', icon: Star },
    { id: 'portfolio', name: 'Portfolio Tracker', description: 'Portfolio performance and allocation tracking', icon: Briefcase },
    { id: 'chart', name: 'Price Chart', description: 'Real-time price charts with technical indicators', icon: BarChart3 },
    { id: 'sentiment', name: 'Sentiment Analysis', description: 'AI-powered sentiment analysis for stocks', icon: TrendingUp },
    { id: 'stats', name: 'Statistics', description: 'Custom statistics and KPI widgets', icon: BarChart3 },
    { id: 'market-data', name: 'Market Data', description: 'Real-time market data grid', icon: TrendingUp }
  ]

  const handleAdd = () => {
    const widgetConfig = {
      title: config.title || `New ${selectedType}`,
      ...config
    }
    onAdd(selectedType, widgetConfig)
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Add Widget</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X size={20} />
          </button>
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Widget Type</label>
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
            >
              {widgetTypes.map(type => (
                <option key={type.id} value={type.id}>{type.name}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Title</label>
            <input
              type="text"
              value={config.title || ''}
              onChange={(e) => setConfig(prev => ({ ...prev, title: e.target.value }))}
              className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
              placeholder="Widget title"
            />
          </div>
          
          {selectedType === 'chart' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Ticker</label>
              <input
                type="text"
                value={config.ticker || ''}
                onChange={(e) => setConfig(prev => ({ ...prev, ticker: e.target.value.toUpperCase() }))}
                className="w-full p-2 rounded-md bg-gray-800 text-white border border-gray-700"
                placeholder="AAPL"
              />
            </div>
          )}
          
          <div className="flex gap-2 pt-4">
            <button
              onClick={handleAdd}
              className="flex-1 py-2 px-4 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
              Add Widget
            </button>
            <button
              onClick={onClose}
              className="flex-1 py-2 px-4 rounded-md bg-gray-600 text-white hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CustomizableDashboard