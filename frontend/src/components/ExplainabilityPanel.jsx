import React from 'react'
export default function ExplainabilityPanel({ details }){
  if(!details) return null
  const { sentiment_score, forecast_trend_pct, weights } = details
  const contribSent = sentiment_score * weights.sentiment
  const contribFor = Math.min(1, forecast_trend_pct*10) * weights.forecast
  return (
    <div className="p-4 bg-white rounded shadow">
      <h3 className="font-semibold">Why this recommendation?</h3>
      <ul className="mt-2">
        <li>Sentiment contribution: {contribSent.toFixed(3)} (sentiment {sentiment_score.toFixed(2)} × weight {weights.sentiment})</li>
        <li>Forecast contribution: {contribFor.toFixed(3)} (trend {forecast_trend_pct.toFixed(3)} × weight {weights.forecast})</li>
      </ul>
      <p className="mt-2 text-sm text-gray-600">Linear combination transparency. For deeper explainability use SHAP for forecasting features.</p>
    </div>
  )
}
