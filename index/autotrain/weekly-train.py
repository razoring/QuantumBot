import json
import yfinance as yf
import functions as functions
import pandas as pd
import random
import warnings
import math
import copy
from datetime import datetime, timedelta
# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

warnings.simplefilter(action='ignore', category=FutureWarning)
charts = functions.Charts()

def distribute(values:list, error:float, proximity:float):
    offset = error*proximity+0.1*random.random()
    nudged = [max(v+random.uniform(-offset,offset), 0.001) for v in values]
    return [float(v/sum(nudged)) for v in nudged]

symbols = {
    # Mega Cap Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    # Momentum
    "TSLA", "AMD", "COIN", "PLTR", "HOOD", "RIVN", "SMCI",
    # Financials (Cyclical)
    "JPM", "BAC", "V", "BRK-B", "PYPL",
    # Consumer Defensive (Defensive/Low Beta)
    "KO", "PG", "WMT", "MCD", "DG",
    # Healthcare (Defensive)
    "JNJ", "UNH", "PFE",
    # Energy/Industrial (Cyclical)
    "XOM", "CVX", "CAT", "GE", "ENB", "H.TO", "CU.TO",
    # Utility/Real Estate (Interest Rate Sensitive)
    "NEE", "O", "NEM",
    # ETFS (Balanced/Fallback)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "ARKK",
    # Bing Suggestions:
    "TNA", "TZA", "ROKU", "SOFI", 
    # Transport:
    "LMT", "BA", "UPS", "FDX", "GM", "F",
    # Downers: 
    "UPST", "AFRM", "CHGG", "BYND", "VIXY",
    # International:
    "BABA", "TSM", "NIO", "SHOP", "BP", "SHEL", "RIO", "BHP",
    # Commodities:
    "GLD", "SLV", "GDX", "USO", "UNG",
    # Crypto:
    "MARA", "RIOT", "MSTR", "GBTC"
    }

symbols = {"AMD"}

#ranges = ["2023-01-01","2025-11-30"]
ranges = ["2023-01-01","2025-12-30"]

"""
biases: {
    "technology": {
        "weight": [[0.2, 0.2, 0.2, 0.2, 0.2], 0],
        "semiconductors": [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
    },
}
"""

started = datetime.now()
biases:dict[str,list] = {}
for symbol in symbols:
    stock = yf.Ticker(symbol)
    info = stock.info
    sector = info.get("sectorKey", info.get("quoteType", "uncategorized")).lower()
    ind = yf.Industry(info.get("industryKey")).name.lower() if info.get("industryKey") else "unknown"
    history = stock.history(start=datetime.strptime(ranges[0], "%Y-%m-%d")-timedelta(days=730), end=datetime.strptime(ranges[1], "%Y-%m-%d"), interval="1d") # 2018 to give prophet data to base off of
    window = history[ranges[0]:ranges[1]]["Close"] #training window
    window = window.resample("W-FRI").last().dropna()
    if history.empty: break
    print(symbol, sector, ind)

    if sector not in biases: biases[sector] = {"weight":[[0.2, 0.2, 0.2, 0.2, 0.2], 0], ind:[[0.2, 0.2, 0.2, 0.2, 0.2], 0]}
    if ind not in biases[sector]:
        biases[sector][ind] = copy.deepcopy(biases[sector]["weight"])
        biases[sector][ind][1] = 0
    
    bestWeight = biases[sector].get(ind)[0]
    for origin, price in window.items(): #origin = fridays
        print(origin)
        bestError = 9999.0
        bestProx = 0.1
        bestGuess = 0
        trials = 0

        tests:list = distribute(bestWeight,bestError,bestProx)
        bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
        errors = []

        for day, guess in enumerate(charts.projectTestWeek(history=history, weights=bias, today=origin)):
            forward = origin + timedelta(days=day)
            if forward not in window.index: continue  # or interpolate

            actual = float(window[forward])
            errors.append((actual - guess) ** 2)    
        mse = None if len(errors) == 0 else sum(errors)/len(errors)
        print(mse)


weights = open("index/weights.txt","w")
weights.write(f"// {started}:{datetime.now()} ({datetime.now()-started}) \n"+json.dumps(biases))