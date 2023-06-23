import yfinance as yf
import pandas as pd
all=pd.DataFrame(yf.download(tickers=['TSLA', "HMC", "F"],start='2020-01-01',end='2022-01-07',progress=False,))
print(all["Close"]["F"][:2])
    