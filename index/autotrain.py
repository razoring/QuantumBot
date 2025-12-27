import concurrent.futures
import numpy as np
import yfinance as yf
import functions
import pandas as pd
import random
import warnings
# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

warnings.simplefilter(action='ignore', category=FutureWarning)

def distribute(values:list, error:float, proximity:float):
    offset = error*proximity
    nudged = [v + random.uniform(-offset, offset) for v in values]
    nudged = [max(v, 0.001) for v in nudged]
    total = sum(nudged)
    return [float(v/total) for v in nudged]

def process_symbol(symbol, ranges, initial_weights):
    print(symbol)
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    sector = info.get("sector")
    if not sector:
        sector = info.get("quoteType", "Uncategorized")
    
    prices = yf.download(symbol, start=ranges[0], end=ranges[1], progress=False)["Close"]
    if prices.empty:
        return None, None, None
        
    local_charts = functions.Charts()
    symbol_daily_results = []
    
    current_best = initial_weights
    
    history = yf.download(symbol, start="2018-01-01", end="2025-11-30", progress=False)

    for date in prices.index:
        bestError = 9999.0
        bestProx = 0.1
        trials = 0
        bestWeight = current_best

        while trials <= 20:
            tests:list = distribute(bestWeight,bestError,bestProx)
            bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
            try:
                res = local_charts.projectTestDay(ticker=symbol, weights=str(bias), history=history, today=date)
                guess = round(float(res[0]), 2)
                actual = round(float(prices.loc[date]), 2)
                error = abs(actual-guess)
                if error < bestError:
                    bestWeight = tests
                    bestError = error
                    bestProx = bestError*0.01
                print(trials, error, bestError, bestProx, tests, bestWeight)
                if error < 0.05: break
                trials += 1
            except:
                trials += 1
                continue
        
        symbol_daily_results.append(bestWeight)
        current_best = bestWeight
        
    return symbol, sector, symbol_daily_results

if __name__ == "__main__":
    symbols = {"NVDA","META","BYND","CHGG","AAPL","BRK-B","JNJ","AMZN","GOOGL","TSLA","AMD","PLTR","RIVN","COIN","AI","UPST","FSLY","OPEN","HOOD","SPY","QQQ","XLF","XLE","ARKK"}
    ranges = ["2023-01-01","2025-11-30"]
    biases:dict[str,list] = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_symbol, s, ranges, [0.2, 0.2, 0.2, 0.2, 0.2]): s for s in symbols}
        
        for future in concurrent.futures.as_completed(futures):
            symbol, sector, daily_results = future.result()
            
            if not daily_results:
                continue

            if sector not in biases:
                biases[sector] = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]

            for bestWeight in daily_results:
                prevWeight, count = biases[sector]
                cma = [(prevWeight[j]*count+bestWeight[j])/(count + 1) for j in range(len(prevWeight))]
                biases[sector] = [cma, count+1]
                print(biases)

    with open("index/weights.txt", "a") as file:
        for s, val in biases.items():
            file.write(f"{s}: {val}\n")