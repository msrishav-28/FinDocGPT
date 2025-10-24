import React, { useRef, useEffect, useState, useMemo } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
} from 'chart.js'
import { Line, Bar } from 'react-chartjs-2'
import 'chartjs-adapter-date-fns'
import zoomPlugin from 'chartjs-plugin-zoom'
import annotationPlugin from 'chartjs-plugin-annotation'
import { ZoomIn, ZoomOut, RotateCcw, TrendingUp, BarChart3 } from 'lucide-react'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
  zoomPlugin,
  annotationPlugin
)

const AdvancedChart = ({ 
  data, 
  type = 'line', 
  title, 
  height = 400,
  realTimeData = null,
  annotations = [],
  onDataPointClick = null,
  showControls = true,
  zoomEnabled = true
}) => {
  const chartRef = useRef(null)
  const [chartType, setChartType] = useState(type)
  const [isZoomed, setIsZoomed] = useState(false)

  // Merge real-time data with historical data
  const chartData = useMemo(() => {
    if (!data) return null

    let processedData = { ...data }
    
    // If real-time data is provided, merge it
    if (realTimeData && data.datasets) {
      processedData = {
        ...data,
        datasets: data.datasets.map((dataset, index) => {
          if (realTimeData.datasets && realTimeData.datasets[index]) {
            return {
              ...dataset,
              data: [...dataset.data, ...realTimeData.datasets[index].data]
            }
          }
          return dataset
        })
      }
    }

    return processedData
  }, [data, realTimeData])

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#e5e7eb',
          usePointStyle: true,
          padding: 20
        }
      },
      title: {
        display: !!title,
        text: title,
        color: '#f9fafb',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        titleColor: '#f9fafb',
        bodyColor: '#e5e7eb',
        borderColor: 'rgba(59, 130, 246, 0.5)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || ''
            if (label) {
              label += ': '
            }
            if (context.parsed.y !== null) {
              label += new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
              }).format(context.parsed.y)
            }
            return label
          }
        }
      },
      zoom: zoomEnabled ? {
        pan: {
          enabled: true,
          mode: 'x',
          onPanComplete: () => setIsZoomed(true)
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
          onZoomComplete: () => setIsZoomed(true)
        }
      } : undefined,
      annotation: {
        annotations: annotations.reduce((acc, annotation, index) => {
          acc[`annotation${index}`] = {
            type: 'line',
            scaleID: 'x',
            value: annotation.value,
            borderColor: annotation.color || '#ef4444',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
              content: annotation.label,
              enabled: true,
              position: 'start',
              backgroundColor: annotation.color || '#ef4444',
              color: '#ffffff'
            }
          }
          return acc
        }, {})
      }
    },
    scales: {
      x: {
        type: data?.labels?.[0] instanceof Date || typeof data?.labels?.[0] === 'string' && !isNaN(Date.parse(data.labels[0])) ? 'time' : 'category',
        time: {
          displayFormats: {
            hour: 'HH:mm',
            day: 'MMM dd',
            week: 'MMM dd',
            month: 'MMM yyyy'
          }
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)'
        },
        ticks: {
          color: '#9ca3af'
        }
      },
      y: {
        grid: {
          color: 'rgba(75, 85, 99, 0.3)'
        },
        ticks: {
          color: '#9ca3af',
          callback: function(value) {
            return new Intl.NumberFormat('en-US', {
              style: 'currency',
              currency: 'USD',
              minimumFractionDigits: 0,
              maximumFractionDigits: 2
            }).format(value)
          }
        }
      }
    },
    onClick: (event, elements) => {
      if (onDataPointClick && elements.length > 0) {
        const element = elements[0]
        const datasetIndex = element.datasetIndex
        const index = element.index
        const value = chartData.datasets[datasetIndex].data[index]
        const label = chartData.labels[index]
        onDataPointClick({ datasetIndex, index, value, label })
      }
    }
  }), [title, zoomEnabled, annotations, onDataPointClick, chartData, data])

  const resetZoom = () => {
    if (chartRef.current) {
      chartRef.current.resetZoom()
      setIsZoomed(false)
    }
  }

  const zoomIn = () => {
    if (chartRef.current) {
      chartRef.current.zoom(1.1)
      setIsZoomed(true)
    }
  }

  const zoomOut = () => {
    if (chartRef.current) {
      chartRef.current.zoom(0.9)
    }
  }

  if (!chartData) {
    return (
      <div className="flex items-center justify-center h-64 rounded-xl glass border border-white/10">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-gray-400 text-sm">Loading chart data...</p>
        </div>
      </div>
    )
  }

  const ChartComponent = chartType === 'bar' ? Bar : Line

  return (
    <div className="relative">
      {showControls && (
        <div className="absolute top-2 right-2 z-10 flex gap-1">
          <button
            onClick={() => setChartType(chartType === 'line' ? 'bar' : 'line')}
            className="p-2 rounded-md bg-gray-800/80 text-gray-300 hover:text-white hover:bg-gray-700/80 transition-colors"
            title={`Switch to ${chartType === 'line' ? 'bar' : 'line'} chart`}
          >
            {chartType === 'line' ? <BarChart3 size={16} /> : <TrendingUp size={16} />}
          </button>
          {zoomEnabled && (
            <>
              <button
                onClick={zoomIn}
                className="p-2 rounded-md bg-gray-800/80 text-gray-300 hover:text-white hover:bg-gray-700/80 transition-colors"
                title="Zoom in"
              >
                <ZoomIn size={16} />
              </button>
              <button
                onClick={zoomOut}
                className="p-2 rounded-md bg-gray-800/80 text-gray-300 hover:text-white hover:bg-gray-700/80 transition-colors"
                title="Zoom out"
              >
                <ZoomOut size={16} />
              </button>
              {isZoomed && (
                <button
                  onClick={resetZoom}
                  className="p-2 rounded-md bg-gray-800/80 text-gray-300 hover:text-white hover:bg-gray-700/80 transition-colors"
                  title="Reset zoom"
                >
                  <RotateCcw size={16} />
                </button>
              )}
            </>
          )}
        </div>
      )}
      
      <div style={{ height: `${height}px` }} className="rounded-xl glass p-4">
        <ChartComponent
          ref={chartRef}
          data={chartData}
          options={options}
        />
      </div>
    </div>
  )
}

export default AdvancedChart