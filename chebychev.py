import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import entropy

@dataclass
class ChebychevForecast:
    ticker: str
    k: float = 10

    def _forecast(self):
      ticker = yf.Ticker(self.ticker).history(period='max', interval='1m')
      print(f"*****Starting fetching data for {self.ticker}.*****")
      if ticker.shape[0] != 0:
          print(f'*****Successfully fetched {ticker.shape[0]} entries for {self.ticker}.*****')
      ticker['close'] = np.log(ticker['Close'])
      ticker['price_std'] = ticker['close'].rolling(window=5).std().shift(1)
      ticker['price_mu'] = ticker['close'].rolling(window=5).mean().shift(1)
      ticker.dropna(inplace=True)
      chebychev_forecast = ticker[['Close','price_std','price_mu']].dropna()

      chebychev_forecast['close_min'] = np.exp(chebychev_forecast['price_mu']-self.k*chebychev_forecast['price_std'])
      chebychev_forecast['close_max'] = np.exp(chebychev_forecast['price_mu']+self.k*chebychev_forecast['price_std'])
      chebychev_forecast['in_range'] = chebychev_forecast['Close'].between(chebychev_forecast['close_min'], chebychev_forecast['close_max'])
      chebychev_forecast['close_avg'] = (chebychev_forecast['close_min'] + chebychev_forecast['close_max'])/2
      return chebychev_forecast, entropy(chebychev_forecast['close_avg'], chebychev_forecast['Close'])