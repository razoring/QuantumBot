# functions.py
import hashlib
import io
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from datetime import datetime, timedelta
import time
from prophet import Prophet as ph
from pyfonts import set_default_font, load_google_font
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import sys as themes #to suppress warnings
import requests
import ast
from collections import OrderedDict
import threading

#Setup
matplotlib.use("Agg")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
set_default_font(load_google_font("Montserrat", weight="bold"))

class Humanizer:
    @staticmethod
    def suffix(number):
        suffixes = ["", "K", "M", "B", "T", "Q"]
        magnitude = 0
        while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
            magnitude += 1
            number /= 1000
        return f"{round(number, 2)}{suffixes[magnitude]}".replace(".0", "")

    @staticmethod
    def sign(number):
        return "+" + str(number) if number > 0 else str(number)

class yFinanceWrapper:
    def __init__(self, ticker):
        self._symbol = yf.Ticker(ticker=ticker)
        self._fastInfo = self._symbol.get_fast_info()
        self._info = self._symbol.info
        self._calendar = self._symbol.calendar
        self._cachedHistory = None 

    def _get_history(self, period="1y"):
        if self._cachedHistory is None:
            self._cachedHistory = self._symbol.history(period=period)
        return self._cachedHistory

    def getStockInfo(self): return self._info
    def getFastInfo(self): return self._fastInfo
    def getCalendar(self): return self._calendar
    
    def getCurrentPrice(self):
        return self._fastInfo.get("lastPrice", 0)

    def getDayOpen(self):
        return self._fastInfo.get("open", 0)
    
    def getDayClose(self):
        return self._fastInfo.get("previousClose", 0)

    def getPriceChange(self):
        openPrice = self.getDayOpen()
        return ((self.getCurrentPrice() / openPrice) * 100) - 100 if openPrice else 0

    def getDayHigh(self): return self._fastInfo.get("dayHigh", 0)
    def getDayLow(self): return self._fastInfo.get("dayLow", 0)
    def get52wkLow(self): return self._fastInfo.get("yearLow", 0)
    def get52wkHigh(self): return self._fastInfo.get("yearHigh", 0)
    def getVolume(self): return self._fastInfo.get("lastVolume", 0)
    def getAvgVolume(self): return self._info.get("averageVolume", 0)
    def getPERatio(self): return self._info.get("trailingPE", 0)
    def getEPSRatio(self): return self._info.get("trailingEps", 0)
    def getMktCap(self): return self._info.get("marketCap", 0)
    def getBeta(self): return self._info.get("beta", 0)

    def getAnnualYield(self):
        #Yahoo often puts the percentage in 'dividendYield' (0.05) and the dollar amount in 'trailingAnnualDividendRate' (1.50)
        if "dividendYield" in self._info and self._info["dividendYield"] is not None:
            return round(self._info["dividendYield"] * 100, 2)
        
        #Fallback
        rate = self._info.get("trailingAnnualDividendRate")
        price = self.getCurrentPrice()
        if rate and price:
            return round((rate / price) * 100, 2)
        return 0

    def getMonthlyYield(self):
        yields = self.getAnnualYield()
        return round(yields / 12.0, 2) if yields != 0 else 0
    
    def getDividendsPayout(self):
        return self._symbol.dividends if self.getAnnualYield() > 0 else None

    def getExDividendDate(self):
        ts = self._info.get("exDividendDate")
        return str(datetime.fromtimestamp(ts).date()) if ts else "-"
    
    def getPayDate(self):
        return str(self._calendar.get("Dividend Date", "-"))

    def getDividendAmount(self):
        divs = self.getDividendsPayout()
        return divs.iloc[-1] if divs is not None and not divs.empty else 0
    
    def getDividendChange(self):
        divs = self.getDividendsPayout()
        if divs is not None and len(divs) >= 2:
            change = (float(divs.iloc[-1]) / float(divs.iloc[-2]) - 1) * 100
            return f"{round(change, 2)}%"
        return "0%"

class Charts:
    def __init__(self): #ttl = time to live (before expiry)
        self._cache = OrderedDict() # {1746164675.3231642:["D":0.2,"W":0.2,"M":0.2,"Y":0.2]}
        self._thread = threading.Lock()
        self._ttl = 60*60*24 # 60 seconds = 60 minutes = 24 hours before expiry
        self._capacity = 64

    def getBatchForecasts(self, history, configs, today):
        today = datetime.strptime(today, "%Y-%m-%d") if isinstance(today, str) else today
        
        # 1. Establish the absolute end date (last known data point)
        # We assume history is sorted.
        if today not in history.index:
            # Fallback to nearest previous date if 'today' is a weekend/holiday
            locs = history.index.get_indexer([today], method='pad')
            if locs[0] == -1: return None
            lastDate = history.index[locs[0]]
        else:
            lastDate = today

        results = []
        
        for h, settings in configs.items():
            freq = settings[1]
            
            # --- CRITICAL FIX START ---
            # Slice the window distinctively for EACH horizon (h)
            # This ensures the 90-day model differs from the 730-day model
            startDate = lastDate - timedelta(days=int(h))
            window = history[(history.index > startDate) & (history.index <= lastDate)]
            
            # Skip if insufficient data for this specific horizon
            if len(window) < 20: 
                results.append(np.zeros(91)) # Or handle gracefully
                continue
            # --- CRITICAL FIX END ---

            # Create data for Prophet
            data = window.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
            data["ds"] = data["ds"].dt.tz_localize(None)
            
            # Optimization: Unique cache key includes the horizon 'h' and start date
            key = (lastDate.isoformat(), h, freq, window["Close"].iloc[-1])

            with self._thread:
                cached = self._cache.get(key)
            
            # If we need to refit (simplified logic for brevity)
            if cached is None:
                config = ph(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
                try:
                    config.fit(data)
                    future = config.make_future_dataframe(periods=91)
                    fcst = config.predict(future)
                    curve = fcst.tail(91)["yhat"].values
                except Exception:
                    curve = np.full(91, window["Close"].iloc[-1]) # Fallback flat line
                
                with self._thread:
                    self._cache[key] = curve
            else:
                curve = cached

            results.append(curve)

        return np.vstack(results)

    def _prophetBacktest(self, history, lastDate, curPrice, histories, forward=90):
        prophetTrend = None
        prophetSigma = 0
        prophetSum = []

        # ensure histories is a dict
        histories = ast.literal_eval(histories.replace('"', "'")) if isinstance(histories, str) else histories

        for h, nested in histories.items():
            startDate = lastDate - timedelta(days=h)
            window = history[history.index > startDate]
            if len(window) < 20:
                continue

            close = window["Close"].copy()

            key = (
                startDate.isoformat(),
                lastDate.isoformat(),
                nested[1],
                tuple(close.values)
            )

            trend = None
            with self._thread:
                item = self._cache.get(key)
                if item:
                    ts, cached = item
                    # TTL check
                    if not self._ttl or (time.time() - ts) <= self._ttl:
                        # valid cache hit
                        self._cache.move_to_end(key)
                        trend = cached
                    else:
                        # expired
                        del self._cache[key]

            if trend is None:
                data = window.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
                data["ds"] = data["ds"].dt.tz_localize(None)

                config = ph(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True, n_changepoints=50, changepoint_prior_scale=0.5, changepoint_range=0.8, uncertainty_samples=2500) # cpps = 0.05
                config.fit(data)

                future = config.make_future_dataframe(periods=forward, freq=nested[1])
                fcst = config.predict(future)
                trend = fcst.tail(forward + 1)["yhat"].values

                with self._thread:
                    self._cache[key] = (time.time(), trend)
                    self._cache.move_to_end(key)

                    # evict oldest if over capacity
                    while len(self._cache) > self._capacity:
                        self._cache.popitem(last=False)
            prophetSum.append((trend + curPrice - trend[0]) * nested[0])

        if prophetSum:
            prophetTrend = np.sum(prophetSum, axis=0)

        return prophetTrend, prophetSigma
    
    def projectTestDay(self, history, weights, today): #period given in days
        today = datetime.strptime(today, "%Y-%m-%d") if type(today) == str else today
        window = history[(history.index >= today - timedelta(days=730)) & (history.index <= today)]

        curPrice = window["Close"].iloc[-1]
        lastDate = window.index[-1]
        
        prophetTrend,_ = self._prophetBacktest(history=window, lastDate=lastDate, curPrice=curPrice, histories=weights, forward=1)
        if prophetTrend is None: raise ValueError("Prophet generation failed")
        return prophetTrend[1]
    
    def projectTestWeek(self, history, weights, today): #period given in days
        today = datetime.strptime(today, "%Y-%m-%d") if type(today) == str else today
        window = history[(history.index >= today - timedelta(days=730)) & (history.index <= today)]

        curPrice = window["Close"].iloc[-1]
        lastDate = window.index[-1]
        
        prophetTrend,_ = self._prophetBacktest(history=window, lastDate=lastDate, curPrice=curPrice, histories=weights, forward=90)
        if prophetTrend is None: raise ValueError("Prophet generation failed")
        median = prophetTrend.copy()
        median = np.maximum(median, 0.01)
        return median