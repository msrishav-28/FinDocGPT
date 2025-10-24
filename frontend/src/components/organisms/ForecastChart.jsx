import React, { useMemo } from 'react'
import { Line } from 'react-chartjs-2'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

export default function ForecastChart({ data }){
  const prefersReducedMotion = (typeof window !== 'undefined' && window.matchMedia) ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false
  const chartData = useMemo(()=>({
    labels: data.ds,
    datasets: [
      { label: 'Predicted', data: data.yhat, borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.15)', fill: false, tension: 0.25, borderWidth: 2 },
      { label: 'Lower', data: data.yhat_lower, borderColor: '#94a3b8', backgroundColor: 'rgba(148,163,184,0.1)', fill: '+1', tension: 0.25, borderDash: [6,6], borderWidth: 1 },
      { label: 'Upper', data: data.yhat_upper, borderColor: '#94a3b8', backgroundColor: 'rgba(148,163,184,0.1)', fill: '-1', tension: 0.25, borderDash: [6,6], borderWidth: 1 }
    ]
  }), [data])
  const options = useMemo(()=>({
    responsive:true,
    maintainAspectRatio:false,
    animation: prefersReducedMotion ? { duration: 0 } : { duration: 600, easing: 'easeOutCubic' },
    plugins:{ 
      legend:{ position:'top', labels: { color: '#e5e7eb' } }, 
      tooltip:{ mode:'index', intersect:false }
    },
    interaction:{mode:'nearest',axis:'x',intersect:false},
    scales:{
      x: { ticks:{ maxTicksLimit: 10, color: '#cbd5e1' }, grid:{ display:false } },
      y: { ticks:{ color: '#cbd5e1' }, grid:{ color:'rgba(203,213,225,0.25)' } }
    }
  }), [prefersReducedMotion])
  return (
    <div className="p-4 glass rounded-xl shadow-card animate-chart-in hover-lift" role="figure" aria-label="Forecast chart">
      <div style={{height: 360}}>
        <Line data={chartData} options={options} aria-hidden={false} />
      </div>
    </div>
  )
}
