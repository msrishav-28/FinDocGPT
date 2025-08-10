from .sentiment_service import analyze_sentiment
from .forecast_service import forecast_prices

def simple_trend_from_forecast(df_forecast):
    if len(df_forecast) < 2:
        return 0.0
    first = df_forecast.iloc[-2]['yhat']
    last = df_forecast.iloc[-1]['yhat']
    try:
        pct = (last - first) / first if first!=0 else 0.0
    except Exception:
        pct = 0.0
    return pct

def recommend(ticker: str, doc_id: str):
    s = analyze_sentiment(doc_id)
    df = forecast_prices(ticker, periods=7)
    trend = simple_trend_from_forecast(df)
    sentiment_weight = 0.6
    forecast_weight = 0.4
    sentiment_score = max(min(s,1), -1)
    forecast_score = max(min(trend*10, 1), -1)
    combined = sentiment_weight*sentiment_score + forecast_weight*forecast_score
    if combined > 0.2:
        rec = "BUY"
    elif combined < -0.2:
        rec = "SELL"
    else:
        rec = "HOLD"
    explanation = {
        'recommendation': rec,
        'score': combined,
        'details': {
            'sentiment_score': sentiment_score,
            'forecast_trend_pct': float(trend),
            'weights': {'sentiment': sentiment_weight, 'forecast': forecast_weight}
        },
        'reasoning_text': f"Sentiment was {sentiment_score:.2f} and forecast trend {trend:.3f}. Weighted combination -> {combined:.3f}."
    }
    return explanation
