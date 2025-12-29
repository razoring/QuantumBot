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

import yfinance as yf
import functions as functions
import pandas as pd
import random
import warnings
from datetime import datetime, timedelta
# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

warnings.simplefilter(action='ignore', category=FutureWarning)
charts = functions.Charts()

def distribute(values:list, error:float, proximity:float):
    offset = error*proximity+0.1*random.random()
    nudged = [max(v+random.uniform(-offset,offset), 0.001) for v in values]
    return [float(v/sum(nudged)) for v in nudged]

ranges = ["2023-01-01","2025-11-30"]
dates = pd.date_range(start=ranges[0], end=ranges[1])

biases:dict[str,list] = {}
for symbol in symbols:
    print(symbol)
    stock = yf.Ticker(symbol)
    info = stock.info
    sector = info.get("sectorKey", info.get("quoteType", "Uncategorized"))
    history = stock.history(start=datetime.strptime(ranges[0], "%Y-%m-%d")-timedelta(days=365), end=datetime.strptime(ranges[1], "%Y-%m-%d"), interval="1d") # 2018 to give prophet data to base off of
    window = history[ranges[0]:ranges[1]]["Close"] #training window
    if history.empty: break

    if sector not in biases:
        biases[sector] = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
    
    bestWeight = biases[sector][0]

    for i, date in enumerate(window.index):
        print(date)
        bestError = 9999.0
        bestProx = 0.1
        trials = 0

        while trials <= 30:
            tests:list = distribute(bestWeight,bestError,bestProx)
            bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
            guess = round(charts.projectTestDay(history=history, weights=str(bias), today=date),2)
            actual = round(round(float(window[date]),2), 2)
            error = abs(actual-guess)
            if error < bestError:
                bestWeight = tests
                bestError = error
                bestProx = bestError*0.02
            #print(trials, guess, actual, error, bestError, bestProx, bestWeight)
            if error <= (actual*0.01)**1.5: break
            trials += 1

        prevWeight, count = biases[sector]
        #ema = [prevWeight[j] * (1 - 0.05) + bestWeight[j]*0.05 for j in range(len(prevWeight))]
        cma = [(prevWeight[j]*count+bestWeight[j])/(count + 1) for j in range(len(prevWeight))]
        biases[sector] = [cma,count+1]
        print(biases)

for s, val in biases.items():
    print(f"{s}: {val}")