symbols = {"NVDA"} # diverse data

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
    offset = error*proximity
    nudged = [v + random.uniform(-offset, offset) for v in values]
    nudged = [max(v, 0.001) for v in nudged]
    total = sum(nudged)
    return [float(v/total) for v in nudged]

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

        while trials <= 50:
            tests:list = distribute(bestWeight,bestError,bestProx)
            bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
            guess = round(charts.projectTestDay(history=history, weights=str(bias), today=date),2)
            actual = round(round(float(window[date]),2), 2)
            error = abs(actual-guess)
            if error < bestError:
                bestWeight = tests
                bestError = error
                bestProx = bestError*0.01
            print(trials, guess, actual, error, bestError, bestProx)
            if error < 0.05: break
            trials += 1

        prevWeight, count = biases[sector]
        cma = [(prevWeight[j]*count+bestWeight[j])/(count + 1) for j in range(len(prevWeight))]
        biases[sector] = [cma,count+1]
        print(biases)

for s, val in biases.items():
    print(f"{s}: {val}")