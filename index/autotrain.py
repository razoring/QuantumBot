symbols = {"NVDA","META","BYND","CHGG","AAPL","BRK-B","JNJ","AMZN","GOOGL","TSLA","AMD","PLTR","RIVN","COIN","AI","UPST","FSLY","OPEN","HOOD","SPY","QQQ","XLF","XLE","ARKK","SPY"} # diverse data

import yfinance as yf
import functions
import pandas as pd
import random
import warnings
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
    sector = yf.Sector(yf.Ticker(symbol).info.get("sectorKey", "Unknown"))
    if sector != "Unknown":
        sector = sector.ticker.info["displayName"]
    prices = yf.download(symbol, start=ranges[0], end=ranges[1], progress=False)["Close"]
    if sector not in biases:
        biases[sector] = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
    
    bestWeight = biases[sector][0]

    for i, date in enumerate(dates):
        bestError = 9999.0
        bestProx = 0.1
        trials = 0

        while trials <= 50:
            tests:list = distribute(bestWeight,bestError,bestProx)
            bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
            guess = round(charts.projectTestDay(ticker=symbol, weights=str(bias), today=date),2)
            actual = round(prices.iloc[i][0],2)
            error = abs(actual-guess)
            if error < bestError:
                bestWeight = tests
                bestError = error
                bestProx = bestError*0.01
            #print(trials, error, bestError, bestProx, tests, bestWeight)
            trials += 1

        prevWeight, count = biases[sector]
        cma = [(prevWeight[j]*count+bestWeight[j])/(count + 1) for j in range(len(prevWeight))]
        biases[sector] = [cma,count+1]
        #print(biases)

with open("index/weights.txt", "a") as file:
    for s, val in biases.items():
        file.write(f"{s}: {val}\n")
    file.close()