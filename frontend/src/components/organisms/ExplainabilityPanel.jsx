import React from 'react'
export default function ExplainabilityPanel({ details }){
  if(!details) return null
  const { sentiment_score, forecast_trend_pct, weights } = details
  const contribSent = sentiment_score * weights.sentiment
  const trendScaled = Math.min(1, Math.max(-1, forecast_trend_pct*10))
  const contribFor = trendScaled * weights.forecast
  const pill = (value)=> value>0? 'text-emerald-700 bg-emerald-50 border-emerald-200' : value<0? 'text-rose-700 bg-rose-50 border-rose-200' : 'text-gray-700 bg-gray-50 border-gray-200'
  return (
    <section aria-labelledby="explain-title" className="p-5 bg-white rounded-xl shadow-card">
      <h3 id="explain-title" className="font-semibold">Why this recommendation?</h3>
      <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className={`rounded-lg border p-3 ${pill(contribSent)}`}>
          <div className="text-xs uppercase tracking-wide">Sentiment</div>
          <div className="text-sm">Contribution: <strong>{contribSent.toFixed(3)}</strong></div>
          <div className="text-xs">Sentiment {sentiment_score.toFixed(2)} × weight {weights.sentiment}</div>
        </div>
        <div className={`rounded-lg border p-3 ${pill(contribFor)}`}>
          <div className="text-xs uppercase tracking-wide">Forecast</div>
          <div className="text-sm">Contribution: <strong>{contribFor.toFixed(3)}</strong></div>
          <div className="text-xs">Trend {forecast_trend_pct.toFixed(3)} × weight {weights.forecast}</div>
        </div>
      </div>
      <p className="mt-3 text-sm text-gray-600">We combine sentiment and time-series trend with transparent linear weights. Consider augmenting with feature attribution for model-level insights.</p>
    </section>
  )
}
