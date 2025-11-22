import React, { useState, useEffect } from 'react'
import { GitCompare, FileText, TrendingUp, TrendingDown, BarChart3, Eye, Download } from 'lucide-react'
import AdvancedChart from './AdvancedChart'

const DocumentComparison = ({ documents = [] }) => {
  const [selectedDocuments, setSelectedDocuments] = useState([])
  const [comparisonData, setComparisonData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [comparisonType, setComparisonType] = useState('sentiment') // sentiment, metrics, timeline
  const [viewMode, setViewMode] = useState('side-by-side') // side-by-side, overlay, table

  useEffect(() => {
    if (selectedDocuments.length >= 2) {
      performComparison()
    }
  }, [selectedDocuments, comparisonType])

  const performComparison = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/documents/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          document_ids: selectedDocuments.map(doc => doc.documentId),
          comparison_type: comparisonType
        })
      })

      if (response.ok) {
        const data = await response.json()
        setComparisonData(data)
      }
    } catch (error) {
      console.error('Error performing comparison:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleDocumentSelection = (document) => {
    setSelectedDocuments(prev => {
      const isSelected = prev.some(doc => doc.id === document.id)
      if (isSelected) {
        return prev.filter(doc => doc.id !== document.id)
      } else if (prev.length < 4) { // Max 4 documents for comparison
        return [...prev, document]
      }
      return prev
    })
  }

  const formatMetricValue = (value, type) => {
    switch (type) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          notation: 'compact',
          maximumFractionDigits: 1
        }).format(value)
      case 'percentage':
        return `${value.toFixed(1)}%`
      case 'number':
        return new Intl.NumberFormat('en-US', {
          notation: 'compact',
          maximumFractionDigits: 1
        }).format(value)
      default:
        return value
    }
  }

  const getMetricTrend = (current, previous) => {
    if (!previous || previous === 0) return null
    const change = ((current - previous) / previous) * 100
    return {
      value: change,
      isPositive: change >= 0,
      formatted: `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`
    }
  }

  const renderSentimentComparison = () => {
    if (!comparisonData?.sentiment_comparison) return null

    const sentimentData = {
      labels: selectedDocuments.map(doc => doc.name.substring(0, 20) + '...'),
      datasets: [
        {
          label: 'Overall Sentiment',
          data: comparisonData.sentiment_comparison.map(item => item.overall_sentiment),
          backgroundColor: 'rgba(59, 130, 246, 0.6)',
          borderColor: 'rgb(59, 130, 246)',
          borderWidth: 2
        },
        {
          label: 'Management Outlook',
          data: comparisonData.sentiment_comparison.map(item => item.management_sentiment),
          backgroundColor: 'rgba(16, 185, 129, 0.6)',
          borderColor: 'rgb(16, 185, 129)',
          borderWidth: 2
        },
        {
          label: 'Financial Performance',
          data: comparisonData.sentiment_comparison.map(item => item.financial_sentiment),
          backgroundColor: 'rgba(245, 158, 11, 0.6)',
          borderColor: 'rgb(245, 158, 11)',
          borderWidth: 2
        }
      ]
    }

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="p-4 rounded-xl glass">
            <h4 className="text-lg font-semibold text-white mb-4">Sentiment Comparison</h4>
            <AdvancedChart
              data={sentimentData}
              type="bar"
              height={300}
              showControls={false}
            />
          </div>
          
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-white">Sentiment Analysis</h4>
            {comparisonData.sentiment_comparison.map((item, index) => (
              <div key={index} className="p-4 rounded-xl glass">
                <div className="flex items-center justify-between mb-3">
                  <h5 className="font-medium text-white">
                    {selectedDocuments[index]?.name.substring(0, 30)}...
                  </h5>
                  <span className={`text-sm px-2 py-1 rounded ${
                    item.overall_sentiment > 0.1 ? 'bg-green-900/30 text-green-300' :
                    item.overall_sentiment < -0.1 ? 'bg-red-900/30 text-red-300' :
                    'bg-gray-900/30 text-gray-300'
                  }`}>
                    {item.overall_sentiment > 0.1 ? 'Positive' :
                     item.overall_sentiment < -0.1 ? 'Negative' : 'Neutral'}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Overall:</span>
                    <span className="text-white">{item.overall_sentiment.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Management:</span>
                    <span className="text-white">{item.management_sentiment.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Financial:</span>
                    <span className="text-white">{item.financial_sentiment.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderMetricsComparison = () => {
    if (!comparisonData?.metrics_comparison) return null

    return (
      <div className="space-y-6">
        <div className="overflow-x-auto">
          <table className="w-full rounded-xl glass overflow-hidden">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="text-left p-4 text-sm font-medium text-gray-300">Metric</th>
                {selectedDocuments.map((doc, index) => (
                  <th key={index} className="text-right p-4 text-sm font-medium text-gray-300">
                    {doc.name.substring(0, 20)}...
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonData.metrics_comparison.map((metric, metricIndex) => (
                <tr key={metricIndex} className="border-t border-white/10">
                  <td className="p-4 font-medium text-white">{metric.name}</td>
                  {metric.values.map((value, valueIndex) => {
                    const trend = valueIndex > 0 ? getMetricTrend(value, metric.values[valueIndex - 1]) : null
                    return (
                      <td key={valueIndex} className="p-4 text-right">
                        <div className="text-white font-semibold">
                          {formatMetricValue(value, metric.type)}
                        </div>
                        {trend && (
                          <div className={`text-sm flex items-center justify-end gap-1 ${
                            trend.isPositive ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {trend.isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                            {trend.formatted}
                          </div>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderTimelineComparison = () => {
    if (!comparisonData?.timeline_comparison) return null

    const timelineData = {
      labels: comparisonData.timeline_comparison.map(item => 
        new Date(item.date).toLocaleDateString()
      ),
      datasets: selectedDocuments.map((doc, index) => ({
        label: doc.name.substring(0, 20) + '...',
        data: comparisonData.timeline_comparison.map(item => item.values[index]),
        borderColor: `hsl(${index * 60}, 70%, 50%)`,
        backgroundColor: `hsla(${index * 60}, 70%, 50%, 0.1)`,
        borderWidth: 2,
        fill: false,
        tension: 0.1
      }))
    }

    return (
      <div className="p-4 rounded-xl glass">
        <h4 className="text-lg font-semibold text-white mb-4">Timeline Comparison</h4>
        <AdvancedChart
          data={timelineData}
          type="line"
          height={400}
          showControls={true}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Document Selection */}
      <div className="p-4 rounded-xl glass">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <GitCompare size={20} />
            Document Comparison
          </h3>
          <span className="text-sm text-gray-400">
            Select 2-4 documents to compare
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {documents.map(document => (
            <div
              key={document.id}
              onClick={() => toggleDocumentSelection(document)}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedDocuments.some(doc => doc.id === document.id)
                  ? 'border-blue-500 bg-blue-900/20'
                  : 'border-gray-600 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center gap-3">
                <FileText className="text-blue-400" size={20} />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-white truncate">{document.name}</p>
                  <p className="text-sm text-gray-400">
                    {new Date(document.uploadedAt).toLocaleDateString()}
                  </p>
                </div>
                {selectedDocuments.some(doc => doc.id === document.id) && (
                  <div className="w-4 h-4 rounded-full bg-blue-500 flex items-center justify-center">
                    <div className="w-2 h-2 rounded-full bg-white" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Comparison Controls */}
      {selectedDocuments.length >= 2 && (
        <div className="flex items-center justify-between p-4 rounded-xl glass">
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400">Comparison Type:</span>
            <select
              value={comparisonType}
              onChange={(e) => setComparisonType(e.target.value)}
              className="px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
            >
              <option value="sentiment">Sentiment Analysis</option>
              <option value="metrics">Financial Metrics</option>
              <option value="timeline">Timeline Comparison</option>
            </select>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400">View Mode:</span>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value)}
              className="px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:border-blue-500 focus:outline-none"
            >
              <option value="side-by-side">Side by Side</option>
              <option value="overlay">Overlay</option>
              <option value="table">Table View</option>
            </select>
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {selectedDocuments.length >= 2 && (
        <div className="space-y-6">
          {loading ? (
            <div className="flex items-center justify-center p-8 rounded-xl glass">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-3" />
              <span className="text-gray-400">Analyzing documents...</span>
            </div>
          ) : comparisonData ? (
            <div>
              {comparisonType === 'sentiment' && renderSentimentComparison()}
              {comparisonType === 'metrics' && renderMetricsComparison()}
              {comparisonType === 'timeline' && renderTimelineComparison()}
            </div>
          ) : (
            <div className="text-center p-8 rounded-xl glass">
              <BarChart3 className="mx-auto text-gray-400 mb-4" size={48} />
              <p className="text-gray-400">
                Select documents and choose a comparison type to begin analysis
              </p>
            </div>
          )}
        </div>
      )}

      {/* Selected Documents Summary */}
      {selectedDocuments.length > 0 && (
        <div className="p-4 rounded-xl glass">
          <h4 className="text-sm font-semibold text-gray-300 mb-3">
            Selected Documents ({selectedDocuments.length})
          </h4>
          <div className="space-y-2">
            {selectedDocuments.map((doc, index) => (
              <div key={doc.id} className="flex items-center justify-between p-2 rounded bg-gray-800/50">
                <div className="flex items-center gap-2">
                  <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                    {index + 1}
                  </span>
                  <span className="text-sm text-white truncate">{doc.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="p-1 text-gray-400 hover:text-blue-400 transition-colors"
                    title="View document"
                  >
                    <Eye size={14} />
                  </button>
                  <button
                    onClick={() => toggleDocumentSelection(doc)}
                    className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                    title="Remove from comparison"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default DocumentComparison