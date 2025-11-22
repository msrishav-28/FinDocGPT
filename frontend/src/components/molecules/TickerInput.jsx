import React from 'react'
import { Input, Button } from '../atoms'

const TickerInput = ({ 
  ticker, 
  onTickerChange, 
  onForecast, 
  onRecommendation, 
  loading = { forecast: false, rec: false } 
}) => {
  const disabled = loading.forecast || loading.rec

  return (
    <div className="flex items-center gap-2">
      <Input
        value={ticker}
        onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
        placeholder="AAPL"
        className="w-36"
        aria-label="Ticker symbol"
      />
      <Button 
        onClick={onForecast} 
        disabled={disabled}
        loading={loading.forecast}
        variant="primary"
      >
        {loading.forecast ? 'Running…' : 'Run Forecast'}
      </Button>
      <Button 
        onClick={onRecommendation} 
        disabled={disabled}
        loading={loading.rec}
        variant="secondary"
      >
        {loading.rec ? 'Computing…' : 'Get Recommendation'}
      </Button>
    </div>
  )
}

export default TickerInput