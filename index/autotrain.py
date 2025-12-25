symbols = {"NVDA","META","BYND","CHGG","AAPL","BRK-B","JNJ","AMZN","GOOGL","TSLA","AMD","PLTR","RIVN","COIN","AI","UPST","FSLY","OPEN","HOOD","SPY","QQQ","XLF","XLE","ARKK","SPY"} # diverse data

import yfinance as yf
import functions
import pandas as pd
from PIL import Image
from IPython.display import display
import random
from datetime import datetime
# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

def distribute(l:list):
    x = len(l)
    r = [random.random() for _ in range(x)]
    m = [l[i] + r[i] for i in range(x)]
    s = sum(m)
    m = [v / s for v in m]
    return m

ranges = ["2022-01-01","2025-11-30"]
dates = pd.date_range(start=ranges[0], end=ranges[1])
volatileWeights = []
stableWeights = []
for symbol in symbols:
    beta = yf.Ticker(symbol).info.get("beta",0)
    prices = yf.download(symbol, start=ranges[0], end=ranges[1], progress=False)["Close"]
    weight = [0.005,0.01,0.485,0.49,0.01] #"duration":[weight,freq] - freq in seconds
    
    for i, date in enumerate(dates):
        realistic = False
        while not realistic:
            tests:list = distribute(weight)
            bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
            charts = functions.Charts()
            guess = round(charts.projectTestDay(ticker=symbol, weights=str(bias), today=date),2)
            price = round(prices.iloc[i][0],2)
            print(symbol, date, guess, price)
            if guess == price:
                if beta > 1:
                    volatileWeights = volatileWeights.append(tests)
                else:
                    stableWeights = stableWeights.append(tests)
                realistic = True