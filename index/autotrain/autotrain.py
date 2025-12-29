import yfinance as yf
import functions as functions
import pandas as pd
import random
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math
import json

warnings.simplefilter(action='ignore', category=FutureWarning)
charts = functions.Charts()

def distribute(values: list, err: float, prox: float):
    offset = err * prox + 0.1 * random.random()
    nudged = [max(v + random.uniform(-offset, offset), 0.001) for v in values]
    s = sum(nudged)
    return [float(v / s) for v in nudged]

# config
ranges = ["2023-01-01", "2025-01-01"]
startDate = datetime.strptime(ranges[0], "%Y-%m-%d") - timedelta(days=365)
endDate = datetime.strptime(ranges[1], "%Y-%m-%d")
dates = pd.date_range(start=ranges[0], end=ranges[1])

# thread-safe biases structure:
# biases[sector] = {
#   "sectorWeights": [w1..w5],
#   "sectorCount": n,
#   "industries": { industryName: ([w1..w5], count) }
# }
biases = {}
biasLock = threading.Lock()

symbols = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "AMD", "COIN", "PLTR", "HOOD", "RIVN", "SMCI",
    "JPM", "BAC", "V", "BRK-B", "PYPL",
    "KO", "PG", "WMT", "MCD", "DG", "DE", "LOW",
    "JNJ", "UNH", "PFE",
    "XOM", "CVX", "CAT", "GE", "ENB", "H.TO", "CU.TO",
    "NEE", "O", "NEM", "TLT", "IEF", "HYG", "LQD",
    "SPY", "QQQ", "IWM", "XLF", "XLE", "ARKK",
    "TNA", "TZA", "ROKU", "SOFI",
    "LMT", "BA", "UPS", "FDX", "GM", "F",                       
    "UPST", "AFRM", "CHGG", "BYND", "VIXY",
    "BABA", "TSM", "NIO", "SHOP", "BP", "SHEL", "RIO", "BHP",
    "GLD", "SLV", "GDX", "USO", "UNG",
    "MARA", "RIOT", "MSTR", "GBTC"
}
symbols = {"AAPL"}

def saveBiases(biases, path="biases.json"):
    out = {}

    for sec, secData in biases.items():
        secWeights = secData["sectorWeights"]
        secCount = secData["sectorCount"]

        inds = {}
        for ind, (w, c) in secData["industries"].items():
            inds[ind] = {
                "weights": w,
                "count": c
            }

        out[sec] = {
            "sectorWeights": secWeights,
            "sectorCount": secCount,
            "industries": inds
        }

    with open(path, "w") as f:
        json.dump(out, f, indent=4)

def loadBiases(path="biases.json"):
    with open(path, "r") as f:
        return json.load(f)
#biases = loadBiases()

def initBiasIfMissing(sec: str, ind: str):
    with biasLock:
        if sec not in biases:
            biases[sec] = {
                "sectorWeights": [0.2, 0.2, 0.2, 0.2, 0.2],
                "sectorCount": 0,
                "industries": {}
            }
        if ind not in biases[sec]["industries"]:
            biases[sec]["industries"][ind] = ([0.2, 0.2, 0.2, 0.2, 0.2], 0)

def updateBiases(sec: str, ind: str, bestWeight: list):
    with biasLock:
        secEntry = biases[sec]
        prevIndW, indCount = secEntry["industries"][ind]
        newIndCount = indCount + 1
        newIndW = [prevIndW[j] * (1 - 0.05) + bestWeight[j]*0.05 for j in range(len(prevIndW))]  #ema
        #newIndW = [(prevIndW[j] * indCount + bestWeight[j]) / newIndCount for j in range(len(prevIndW))] #cma
        secEntry["industries"][ind] = (newIndW, newIndCount)

        prevSecW = secEntry["sectorWeights"]
        secCount = secEntry["sectorCount"]
        newSecCount = secCount + 1
        newSecW = [prevSecW[j] * (1 - 0.05) + bestWeight[j]*0.05 for j in range(len(prevSecW))]  #ema
        #newSecW = [(prevSecW[j] * secCount + bestWeight[j]) / newSecCount for j in range(len(prevSecW))] #cma
        secEntry["sectorWeights"] = newSecW
        secEntry["sectorCount"] = newSecCount

def processSymbol(sym: str):
    try:
        print(sym)
        stock = yf.Ticker(sym)
        info = stock.info or {}
        sec = info.get("sectorKey", info.get("quoteType", "Uncategorized")) or "Uncategorized"
        ind = info.get("industryKey", "Unknown") or "Unknown"

        # fetch history once
        hist = stock.history(start=startDate, end=endDate, interval="1d")
        if hist.empty:
            print(f"no history for {sym}")
            return

        # training window (Close series)
        try:
            window = hist[ranges[0]:ranges[1]]["Close"]
        except Exception:
            window = hist["Close"].loc[ranges[0]:ranges[1]]

        if window.empty:
            print(f"no window for {sym}")
            return

        initBiasIfMissing(sec, ind)

        # start from sector-level weights (thread-safe read)
        with biasLock:
            baseWeight = list(biases[sec]["sectorWeights"])

        # iterate dates
        for i, dt in enumerate(window.index):
            # small guard: ensure dt is within hist index
            if dt not in hist.index:
                continue

            bestWeight = list(baseWeight)
            bestError = float("inf")
            bestProx = 0.1
            trials = 0

            # prepare history slice for this date
            # use iloc slicing for speed: find position
            try:
                pos = hist.index.get_loc(dt)
            except KeyError:
                continue
            startPos = max(0, pos - 365)
            windowSlice = hist.iloc[startPos:pos + 1]
            if len(windowSlice) < 20:
                continue

            # inner optimization loop
            while trials <= 30:
                tests = distribute(bestWeight, bestError, bestProx)
                biasDict = {
                    90: [tests[0], "ME"],
                    180: [tests[1], "ME"],
                    365: [tests[2], "D"],
                    730: [tests[3], "W"],
                    1825: [tests[4], "YS"]
                }
                try:
                    guess = charts.projectTestDay(history=windowSlice, weights=biasDict, today=dt)
                    if guess is None:
                        raise ValueError("prophet failed")
                    guess = round(float(guess), 2)
                except Exception:
                    # if prophet fails, break and skip this date
                    break

                actual = round(float(window[dt]), 2)
                err = abs(actual - guess)
                if err < bestError:
                    bestWeight = tests
                    bestError = err
                    bestProx = max(bestError * 0.02, 1e-6)
                # early stop: relative threshold scaled by price
                if actual > 0 and err <= math.pow(actual * 0.01, 1.5):
                    break
                trials += 1

            print(trials, actual, err, bestWeight)
            # update biases with the bestWeight found for this date
            updateBiases(sec, ind, bestWeight)

    except Exception as e:
        print(f"error {sym}: {e}")

def main():
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = [exe.submit(processSymbol, s) for s in symbols]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("worker error:", e)
    saveBiases(biases=biases,path="weights.json")
    print(json.dumps(loadBiases("weights.json"), indent=4))

if __name__ == "__main__":
    main()
