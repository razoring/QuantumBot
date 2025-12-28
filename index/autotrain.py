import concurrent.futures
import numpy as np
import yfinance as yf
import functions
import random
import warnings
import sys
import logging

# loop through symbols, use predictTest, use frequencies as seconds, use 1mo,3mo,6mo,1y,2y,5y as range

warnings.simplefilter(action='ignore', category=FutureWarning)

def distribute(values:list, error:float, proximity:float):
    offset = error*proximity
    nudged = [v + random.uniform(-offset, offset) for v in values]
    nudged = [max(v, 0.001) for v in nudged]
    total = sum(nudged)
    return [float(v/total) for v in nudged]

def processSymbol(symbol, ranges, weight):
    logging.info(f"STATUS: {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        sector = info.get("sector")
        if not sector:
            sector = info.get("quoteType", "Uncategorized")
        
        history = yf.download(symbol, start="2018-01-01", end="2025-11-30", progress=False)
        
        if history.empty:
            return None, None, None

        prices = history["Close"][(history.index >= ranges[0]) & (history.index <= ranges[1])]
        
        if prices.empty:
            return None, None, None
            
        charts = functions.Charts()
        daily = []
        
        best = weight
        
        for date in prices.index:
            bestError = 9999.0
            bestProx = 0.1
            trials = 0
            bestWeight = best

            max_trials = 100 if len(daily) == 0 else 20

            while trials <= max_trials:
                tests:list = distribute(bestWeight,bestError,bestProx)
                bias = {90:[tests[0], "ME"], 180:[tests[1], "ME"], 365:[tests[2], "D"], 730:[tests[3], "W"], 1825:[tests[4], "YS"]}
                try:
                    results = charts.projectTestDay(ticker=symbol, weights=str(bias), history=history, today=date)
                    
                    guess = round(float(results), 2)
                    actual = round(float(prices.loc[date]), 2)
                    
                    error = abs(actual-guess)
                    if error < bestError:
                        bestWeight = tests
                        bestError = error
                        bestProx = bestError*0.01
                    
                    if error < 0.05: break
                    trials += 1
                except Exception:
                    trials += 1
                    continue
            
            daily.append(bestWeight)
            best = bestWeight
        return symbol, sector, daily
    except Exception as e:
        logging.info(f"CRITICAL ERROR for {symbol}: {e}")
        return None, None, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

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
    "JNJ", "UNH", "PFE", "PFE",
    # Energy/Industrial (Cyclical)
    "XOM", "CVX", "CAT", "GE", "ENB", "H.TO", "CU.TO",
    # Utility/Real Estate (Interest Rate Sensitive)
    "NEE", "O", "NEM", "CVX",
    # ETFS (Balanced/Fallback)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "ARKK"}

    ranges = ["2023-01-01","2025-11-30"]
    biases:dict[str,list] = {}

    try:
        dummy = yf.Ticker("SPY")
        _ = dummy.info # Triggers cookie save
        _ = dummy.history(period="1d") # Triggers data cookie save
        logging.info("STATUS: Cache Primed. Starting Parallel Processing")
    except Exception as e:
        logging.info(f"ERROR: Cache priming failed, proceeding anyway: {e}")
        sys.exit(1)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(processSymbol, s, ranges, [0.2, 0.2, 0.2, 0.2, 0.2]): s for s in symbols}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                symbol, sector, results = future.result()
                
                if not results:
                    continue

                if sector not in biases:
                    biases[sector] = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]

                for bestWeight in results:
                    data = biases[sector]
                    avg = np.array(data[0])
                    n = data[1]
                    
                    cma = (avg * n + np.array(bestWeight))/(n+1)
                    
                    # Save as [list, int]
                    biases[sector] = [cma.tolist(),(n+1)]
            except Exception as exc:
                logging.info(f"ERROR: {exc}")
    logging.info("STATUS: PROCESSING FINISHED")
    with open("data/weights.txt", "w") as file:
        for s, val in biases.items():
            line = f"{s}: {val}"
            file.write(line + "\n")