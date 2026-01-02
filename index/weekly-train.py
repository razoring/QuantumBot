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
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "NFLX", "ORCL", "IBM",
    "TSLA", "AMD", "COIN", "PLTR", "HOOD", "RIVN", "SMCI", "AVGO", "NET", "SNOW", "CRWD", "DDOG", "MDB", "ZS",
    "JPM", "BAC", "V", "BRK-B", "PYPL", "GS", "MS", "WFC", "TD", "RY", "AXP", "AIG", "SCHW", "BTC-USD", "ETH-USD",
    "KO", "PG", "WMT", "MCD", "DG", "DE", "HON", "MMM", "DOW", "FCX", "LIN", "COST", "HD", "LOW", "TGT", "SBUX", "NKE",
    "JNJ", "UNH", "PFE", "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "ENB.TO", "CNQ.TO", "SU.TO", "TRP.TO", "ABX.TO", "WPM.TO", "CP.TO", "CNR.TO", "BCE.TO", "T.TO",
    "XOM", "CVX", "CAT", "GE", "ENB", "H.TO", "CU.TO", "SLB", "HAL", "DVN", "COP",
    "NEE", "O", "NEM", "TLT", "IEF", "HYG", "LQD",
    "SPY", "QQQ", "IWM", "XLF", "XLE", "ARKK",
    "TNA", "TZA", "ROKU", "SOFI", 
    "LMT", "BA", "UPS", "FDX", "GM", "F",
    "UPST", "AFRM", "CHGG", "BYND", "VIXY", "UVXY", "SVXY", "SPXU", "SQQQ"
    "BABA", "TSM", "NIO", "SHOP", "BP", "SHEL", "RIO", "BHP",
    "GLD", "SLV", "GDX", "USO", "UNG",
    "MARA", "RIOT", "MSTR", "BRK-A",
    "^SPX", "^VIX", "EWZ", "EEM", "EWJ", "FXI", "VGK", "EFA", "EWU", "EWG", "EWQ", "EWC"
    }

#symbols = {"NVDA"}

#ranges = ["2023-01-01","2025-11-30"]
train = ["2023-12-31","2024-12-31"]
validation = ["2024-01-01","2025-12-31"]

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
    history = stock.history(start=datetime.strptime(train[0], "%Y-%m-%d")-timedelta(days=730), end=datetime.strptime(validation[1], "%Y-%m-%d"), interval="1d") # 2018 to give prophet data to base off of
    if history.empty: break
    print(symbol, sector, ind)

    if sector not in biases: biases[sector] = {"weight":[[0.2, 0.2, 0.2, 0.2, 0.2], 0], ind:[[0.2, 0.2, 0.2, 0.2, 0.2], 0]}
    if ind not in biases[sector]:
        biases[sector][ind] = copy.deepcopy(biases[sector]["weight"])
        biases[sector][ind][1] = 0
    
    bestWeight = biases[sector].get(ind)[0]
    for i in range(3): #1: generation, #2: validation, #3 test unknown
        print(f'Iteration: {list(["Training","Validation","Testing"])[i]}')
        window = history[(train[0] if i < 2 else validation[0]) : (train[1] if i < 2 else validation[1])]["Close"].dropna()
        daily = window.resample("D").interpolate()
        origins = window.resample("W-FRI").last().dropna()

        for origin, price in origins.items(): #origin = fridays
            bias = {90:[biases[sector][ind][0][0], "ME"], 180:[biases[sector][ind][0][1], "ME"], 365:[biases[sector][ind][0][2], "D"], 730:[biases[sector][ind][0][3], "W"], 1825:[biases[sector][ind][0][4], "YS"]}
            rawCurves = charts.getBatchForecasts(history, bias, origin)
            
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
            #adjustment = max(-0.03*math.sqrt(bestError)+0.06,0) #almost equal bias (bias to correct)
            #adjustment = 0.001/(bestError+0.02)+0.03*bestError # bias to correct and incorrect
            adjustment = 0.003/(bestError+0.05)+0.01*bestError # bias to correct
            #adjustment = 0.05 #equal
            avgInd = [prevInd[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevInd))] #ema
            avgSect = [prevSect[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevSect))] #ema
            biases[sector][ind] = [avgInd,countInd+1]
            biases[sector]["weight"] = [avgSect,countSect+1]
            print(origin.date(), bestError, str(round(adjustment*100,2))+"%", bestWeight)

weights = open("index/weights.txt","w")
weights.write(json.dumps(biases)+f"\n// {started}:{datetime.now()} ({datetime.now()-started})")