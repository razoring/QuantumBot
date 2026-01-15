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
import re
# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

warnings.simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings(): warnings.filterwarnings("ignore", category=RuntimeWarning)
charts = functions.Charts()

symbols = []
#with open("index\modular\symbols.txt", "r") as file: symbols = re.sub(r"/\*.*?\*/", "", file.read().replace("\n","").strip().replace(" ",""), flags=re.DOTALL)[:len(file.read())-1].split(",")
with open("index\modular\symbols-test.txt", "r") as file: symbols = re.sub(r"/\*.*?\*/", "", file.read().replace("\n","").strip().replace(" ",""), flags=re.DOTALL)[:len(file.read())-1].split(",")
print(symbols)

"""
biases: {
    "global": [[0.2, 0.2, 0.2, 0.2, 0.2], 0] # ..., x] <-- count of items processed
    "technology": {
        "weight": [[0.2, 0.2, 0.2, 0.2, 0.2], 0],
        "semiconductors": [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
    },
}
"""

#ranges = ["2023-01-01","2025-11-30"]
train = ["2020-01-01","2023-12-31"]
valid = ["2024-01-01","2024-12-31"]
tests = ["2025-01-01","2025-12-31"]

started = datetime.now()
#biases:dict[str,list] = {} #start fresh
with open("index\modular\weights.txt","r") as file: biases = json.loads(file.readlines()[0])
for symbol in symbols:
    stock = yf.Ticker(symbol)
    info = stock.info
    sector = info.get("sectorKey", info.get("quoteType", "uncategorized")).lower()
    ind = yf.Industry(info.get("industryKey")).name.lower() if info.get("industryKey") else str.lower(info.get("category")) if info.get("category") else "unknown"
    history = stock.history(start=datetime.strptime(train[0], "%Y-%m-%d")-timedelta(days=730), end=datetime.strptime(tests[1], "%Y-%m-%d"), interval="1d") # 2018 to give prophet data to base off of
    if history.empty: continue
    print(symbol, sector, ind)

    if sector not in biases: biases[sector] = {"weight":[[0.2, 0.2, 0.2, 0.2, 0.2], 0], ind:[[0.2, 0.2, 0.2, 0.2, 0.2], 0]}
    if ind not in biases[sector]:
        biases[sector][ind] = copy.deepcopy(biases[sector]["weight"])
        biases[sector][ind][1] = 0
    
    bestWeight = biases[sector].get(ind)[0]
    for i in range(3): #1: generation, #2: validation, #3 test unknown
        print(f'Iteration: {list(["Training","Validation","Testing"])[i]}')
        window = history[(train[0] if i < 2 else tests[0]) : (train[1] if i < 2 else tests[1])]
        daily = window.resample("D").interpolate()
        if daily.index.tz is not None: daily.index = daily.index.tz_convert("America/New_York").tz_localize(None)
        origins = window["Close"].resample("W-FRI").last().dropna()

        for origin, price in origins.items(): #origin = fridays
            bias = {90:[biases[sector][ind][0][0], "ME"], 180:[biases[sector][ind][0][1], "ME"], 365:[biases[sector][ind][0][2], "D"], 730:[biases[sector][ind][0][3], "W"], 1825:[biases[sector][ind][0][4], "YS"]}
            rawCurves = charts.getBatchForecasts(window, bias, origin)
            
            if rawCurves is None: continue
            targetDates = [origin + timedelta(days=i) for i in range(91)]
            validIndices = []
            actuals = []
            
            for i, date in enumerate(targetDates):
                d = date.tz_convert("America/New_York").tz_localize(None) if date.tzinfo is not None else date
                if d in daily.index:
                    validIndices.append(i)
                    actuals.append(float(daily.loc[d, "Close"]))
            
            if not validIndices: continue
            matrix = rawCurves[:, validIndices]
            targets = np.array(actuals)

            tune = 0.01  # tune
            def smapeLoss(w):
                predictions = np.dot(w, matrix)
                denom = (np.abs(targets) + np.abs(predictions))
                diff = 2 * np.abs(predictions - targets) / (denom + 1e-8)
                smape = np.mean(diff)
                #penalty = tune * np.sum(w * np.log((w + 1e-8) * 5))
                start = targets[0] if len(targets) > 0 else 0
                end = predictions[-1]
                penalty = 0
                if start > 0:
                    change = abs((end-start)/start)
                    threshold = 0.25
                    if change > threshold:
                        penalty = (change-threshold)*2
                return smape + penalty

            constraints = ({'type': 'eq', 'fun': lambda w:  np.sum(w) - 1.0})
            bounds = ((0.0,1.0),(0.0,1.0),(0.0,1.0),(0.05,1.0),(0.05,1.0)) #(min,max) weight for bounds
            initGuess = np.array(biases[sector].get(ind)[0])
            initGuess = initGuess / np.sum(initGuess)

            res = minimize(smapeLoss, initGuess, method='SLSQP', bounds=bounds, constraints=constraints)
            bestWeight = res.x.tolist()
            bestError = res.fun
            
            fullPreds = np.dot(bestWeight, rawCurves) 
            bestGuess = fullPreds[0] 
            actual = targets[0] if len(targets) > 0 else 0

            prevInd, countInd = biases[sector][ind]
            prevSect, countSect = biases[sector]["weight"]
            #adjustment = max(-0.03*math.sqrt(bestError)+0.06,0) #almost equal bias (bias to correct)
            #adjustment = 0.001/(bestError+0.02)+0.03*bestError # bias to correct and incorrect
            #adjustment = 0.003/(bestError+0.05)+0.01*bestError # bias to correct
            adjustment = 0.05 #equal
            #adjustment = 0.02 + (0.1 * min(bestError, 1.0)) # aggressive correction
            avgInd = [prevInd[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevInd))] #ema
            avgSect = [prevSect[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevSect))] #ema
            biases[sector][ind] = [avgInd,countInd+1]
            biases[sector]["weight"] = [avgSect,countSect+1]
            print(origin.date(), bestError, str(round(adjustment*100,2))+"%", bestWeight)

with open("index\modular\weights.txt","w") as weights: weights.write(json.dumps(biases)+f"\n// {started}:{datetime.now()} ({datetime.now()-started})")