# weekly-train.py
import json
import math
import yfinance as yf
import functions as functions
import pandas as pd
import warnings
from scipy.optimize import minimize
import copy
import numpy as np
from datetime import datetime, timedelta
# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

warnings.simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings(): warnings.filterwarnings("ignore", category=RuntimeWarning)
charts = functions.Charts()

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
ranges = ["2023-12-31","2024-12-31"]

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
    window = history[ranges[0]:ranges[1]]["Close"].dropna()
    daily = window.resample("D").interpolate()
    origins = window.resample("W-FRI").last().dropna()
    if history.empty: break
    print(symbol, sector, ind)

    if sector not in biases: biases[sector] = {"weight":[[0.2, 0.2, 0.2, 0.2, 0.2], 0], ind:[[0.2, 0.2, 0.2, 0.2, 0.2], 0]}
    if ind not in biases[sector]:
        biases[sector][ind] = copy.deepcopy(biases[sector]["weight"])
        biases[sector][ind][1] = 0
    
    bestWeight = biases[sector].get(ind)[0]
    for origin, price in origins.items(): #origin = fridays
        configKeys = [90, 180, 365, 730, 1825] 
        currentBias = {90: [0, "ME"], 180: [0, "ME"], 365: [0, "D"], 730: [0, "W"], 1825: [0, "YS"]}
        rawCurves = charts.getBatchForecasts(history, currentBias, origin)
        
        if rawCurves is None: continue
        targetDates = [origin + timedelta(days=i) for i in range(91)]
        validIndices = []
        actuals = []
        
        for i, date in enumerate(targetDates):
            if date in daily.index:
                validIndices.append(i)
                actuals.append(float(daily[date]))
        
        if not validIndices: continue
        matrix = rawCurves[:, validIndices]
        targets = np.array(actuals)

        def smapeLoss(w):
            preds = np.dot(w, matrix)
            denom = (np.abs(targets) + np.abs(preds))
            diff = 2 * np.abs(preds - targets) / (denom + 1e-8)
            return np.mean(diff)

        cons = ({'type': 'eq', 'fun': lambda w:  np.sum(w) - 1.0})
        bnds = tuple((0.0, 1.0) for _ in range(5))
        initGuess = np.array(biases[sector].get(ind)[0])
        initGuess = initGuess / np.sum(initGuess)

        
        res = minimize(smapeLoss, initGuess, method='SLSQP', bounds=bnds, constraints=cons)
        bestWeight = res.x.tolist()
        bestError = res.fun
        
        fullPreds = np.dot(bestWeight, rawCurves) 
        bestGuess = fullPreds[0] 
        actual = targets[0] if len(targets) > 0 else 0

        prevInd, countInd = biases[sector][ind]
        prevSect, countSect = biases[sector]["weight"]
        adjustment = max(-0.03*math.sqrt(bestError)+0.06,0) #almost equal bias (bias to correct)
        #adjustment = 0.001/(bestError+0.02)+0.03*bestError # bias to correct and incorrect
        #adjustment = 0.003/(bestError+0.05)+0.01*bestError # bias to correct
        #adjustment = 0.05 #equal
        avgInd = [prevInd[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevInd))] #ema
        avgSect = [prevSect[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevSect))] #ema
        biases[sector][ind] = [avgInd,countInd+1]
        biases[sector]["weight"] = [avgSect,countSect+1]
        print(origin.date(), bestError, str(round(adjustment*100,2))+"%", bestWeight)

weights = open("index/weights.txt","w")
weights.write(json.dumps(biases)+f"\n// {started}:{datetime.now()} ({datetime.now()-started})")