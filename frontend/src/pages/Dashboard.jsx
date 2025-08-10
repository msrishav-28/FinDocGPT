import React, { useState } from 'react'
import API from '../api'
import ForecastChart from '../components/ForecastChart'
import ExplainabilityPanel from '../components/ExplainabilityPanel'

export default function Dashboard(){
  const [forecast, setForecast] = useState(null)
  const [rec, setRec] = useState(null)

  async function runTicker(ticker){
    try{
      const resp = await API.get('/forecast', { params: { ticker } })
      setForecast(resp.data)
    }catch(e){
      alert('Forecast failed. Check backend is running and ticker is valid.')
    }
  }

  async function getRecommendation(ticker){
    const form = new FormData(); form.append('ticker', ticker); form.append('doc_id', 'demo_doc')
    const r = await API.post('/recommend', form)
    setRec(r.data)
  }

  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <h2 className="text-lg">Forecast</h2>
        <div className="mb-2">
          <input id="ticker" placeholder="AAPL" className="border p-2 mr-2" />
          <button onClick={()=>runTicker(document.getElementById('ticker').value)} className="px-3 py-1 bg-indigo-600 text-white rounded">Run</button>
        </div>
        {forecast && <ForecastChart data={forecast} />}
      </div>
      <div>
        <h2 className="text-lg">Recommendation</h2>
        <div className="mb-2">
          <input id="rticker" placeholder="AAPL" className="border p-2 mr-2" />
          <button onClick={()=>getRecommendation(document.getElementById('rticker').value)} className="px-3 py-1 bg-green-600 text-white rounded">Get Recommendation</button>
        </div>
        {rec && (
          <div>
            <div className="p-4 rounded shadow bg-white">
              <div className="text-2xl font-bold">{rec.recommendation}</div>
              <div className="text-sm text-gray-600">Score: {rec.score.toFixed(3)}</div>
              <div className="mt-2 text-sm">{rec.reasoning_text}</div>
            </div>
            <div className="mt-4"><ExplainabilityPanel details={rec.details} /></div>
          </div>
        )}
      </div>
    </div>
  )
}
