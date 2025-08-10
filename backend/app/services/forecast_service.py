import yfinance as yf
import pandas as pd
from prophet import Prophet

def forecast_prices(ticker: str, periods: int = 7):
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    data = data.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds','Close':'y'})
    model = Prophet(daily_seasonality=False)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    df = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(periods+30)
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')
    return df
