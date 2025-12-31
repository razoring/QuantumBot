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

symbols = {"NVDA"}

#ranges = ["2023-01-01","2025-11-30"]
ranges = ["2025-01-01","2025-12-30"]
dates = pd.date_range(start=ranges[0], end=ranges[1])

"""
biases: {
    "technology": {
        "weight": [[0.2, 0.2, 0.2, 0.2, 0.2], 0],
        "semiconductors": [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
    },
}
"""

biases:dict[str,list] = {}
for symbol in symbols:
    print(symbol)
    stock = yf.Ticker(symbol)
    info = stock.info
    sector = info.get("sectorKey", info.get("quoteType", "uncategorized")).lower()
    ind = yf.Industry(info.get("industryKey")).name.lower() if info.get("industryKey") else "unknown"
    history = stock.history(start=datetime.strptime(ranges[0], "%Y-%m-%d")-timedelta(days=365), end=datetime.strptime(ranges[1], "%Y-%m-%d"), interval="1d") # 2018 to give prophet data to base off of
    window = history[ranges[0]:ranges[1]]["Close"] #training window
    if history.empty: break

    if sector not in biases: biases[sector] = {"weight":[[0.2, 0.2, 0.2, 0.2, 0.2], 0], ind:[[0.2, 0.2, 0.2, 0.2, 0.2], 0]}
    if ind not in biases[sector]:
        biases[sector][ind] = copy.deepcopy(biases[sector]["weight"])
        biases[sector][ind][1] = 0
    
    bestWeight = biases[sector].get(ind)[0]
    for i, date in enumerate(window.index):
        print(date)
        bestError = 9999.0
        bestProx = 0.1
        bestGuess = 0
        trials = 0

        while trials <= 30:
            tests:list = distribute(bestWeight,bestError,bestProx)
            bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
            guess = round(charts.projectTestDay(history=history, weights=bias, today=date),2)
            actual = round(round(float(window[date]),2), 2)
            error = abs(actual-guess)
            if error < bestError:
                bestWeight = tests
                bestError = error
                bestGuess = guess
                bestProx = bestError*0.02
            print(trials, guess, actual, error, bestError, tests)
            if error <= max(0.04*math.log10(actual),0.001): break # short-circut, early exit
            trials += 1

        prevInd, countInd = biases[sector][ind]
        prevSect, countSect = biases[sector]["weight"]
        #avg = [(prevWeight[j]*count+bestWeight[j])/(count + 1) for j in range(len(prevWeight))] #cma
        avgInd = [prevInd[j] * (1 - 0.05) + bestWeight[j]*0.05 for j in range(len(prevInd))] #ema
        avgSect = [prevSect[j] * (1 - 0.05) + bestWeight[j]*0.05 for j in range(len(prevSect))] #ema
        biases[sector][ind] = [avgInd,countInd+1]
        biases[sector]["weight"] = [avgSect,countSect+1]
        
        print("best:", bestGuess, actual, error, bestError, bestProx, bestWeight)

weights = open("index/weights.txt","w")
weights.write(f"// {datetime.today().date()} \n"+json.dumps(biases))