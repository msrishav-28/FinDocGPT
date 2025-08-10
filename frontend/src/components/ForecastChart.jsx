import React from 'react'
import { Line } from 'react-chartjs-2'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

export default function ForecastChart({ data }){
  const chartData = {
    labels: data.ds,
    datasets: [
      { label: 'Predicted', data: data.yhat, fill: false, tension: 0.2 },
      { label: 'Lower', data: data.yhat_lower, fill: '+1', tension: 0.2, borderDash: [6,6] },
      { label: 'Upper', data: data.yhat_upper, fill: '-1', tension: 0.2, borderDash: [6,6] }
    ]
  }
  const options = { responsive:true, plugins:{ legend:{position:'top'} }, interaction:{mode:'nearest',axis:'x',intersect:false} }
  return <div className="p-4 bg-white rounded shadow"><Line data={chartData} options={options} /></div>
}
