import React, { useMemo, useState } from 'react'
import { forecastService } from '../services'
import { ForecastChart, ExplainabilityPanel } from '../components/organisms'
import { SentimentCard } from '../components/molecules'
import { StatCard, Card } from '../components/atoms'
import { TickerInput } from '../components/molecules'

export default function Dashboard(){
  const [ticker, setTicker] = useState('AAPL')
  const [forecast, setForecast] = useState(null)
  const [rec, setRec] = useState(null)
  const [loading, setLoading] = useState({ forecast: false, rec: false })
  const [error, setError] = useState(null)

  async function runTicker(){
    setError(null); setLoading(l => ({ ...l, forecast: true }))
    try{
      const data = await forecastService.getForecast(ticker)
      setForecast(data)
    }catch(e){
      setError('Forecast failed. Ensure backend is running and ticker is valid.')
    }finally{
      setLoading(l => ({ ...l, forecast: false }))
    }
  }

  async function getRecommendation(){
    setError(null); setLoading(l => ({ ...l, rec: true }))
    try{
      const data = await forecastService.getRecommendation(ticker, 'demo_doc')
      setRec(data)
    }catch(e){
      setError('Recommendation failed. Check backend connectivity.')
    }finally{
      setLoading(l => ({ ...l, rec: false }))
    }
  }

  const disabled = useMemo(()=> loading.forecast || loading.rec, [loading])

  return (
  <div className="space-y-6">
    <Card className="p-5 hover-lift">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Interactive Forecast & Recommendation</h2>
          <p className="text-sm text-gray-400">Enter a ticker and run the modules. Data is fetched live.</p>
        </div>
        <TickerInput
          ticker={ticker}
          onTickerChange={setTicker}
          onForecast={runTicker}
          onRecommendation={getRecommendation}
          loading={loading}
        />
      </div>
      {error && <div role="alert" className="mt-3 rounded-md border border-rose-900/50 bg-rose-900/20 px-3 py-2 text-sm text-rose-200">{error}</div>}
    </Card>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SentimentCard docId="demo_doc" ticker={ticker} />
        <StatCard label="Data horizon" value="~1y" hint="Yahoo Finance daily" tone="brand" />
        <StatCard label="Forecast window" value="7 days" hint="Prophet extrapolation" tone="gray" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Forecast</h3>
          {forecast ? (
            <ForecastChart data={forecast} />
          ) : (
            <Card className="p-8 border border-dashed border-white/10">
              <div className="text-sm text-gray-400">No forecast yet. Run a forecast to visualize predicted prices.</div>
            </Card>
          )}
        </div>
        <div className="lg:col-span-2">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Recommendation</h3>
          {rec ? (
            <div>
              <Card className="p-5 hover-lift">
                <div className="flex items-baseline gap-3">
                  <div className="text-3xl font-bold tracking-tight text-white">{rec.recommendation}</div>
                  <div className="text-xs text-gray-400">Score {Number(rec.score).toFixed(3)}</div>
                </div>
                <p className="mt-2 text-sm text-gray-300">{rec.reasoning_text}</p>
              </Card>
              <div className="mt-4"><ExplainabilityPanel details={rec.details} /></div>
            </div>
          ) : (
            <Card className="p-8 border border-dashed border-white/10">
              <div className="text-sm text-gray-400">No recommendation yet. Click Get Recommendation.</div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
