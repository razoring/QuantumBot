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

class Stamp:
    def __init__(self, name, url, icon):
        self.serverName = name
        self.serverInvite = str(url)
        self.serverIcon = icon

    def _font(self, size: int):
        return ImageFont.truetype(font="index/assets/Montserrat-Bold.ttf", size=size)

    def _rounded(self, image: Image.Image, radius: int) -> Image.Image:
        image = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        rounded = Image.new("RGBA", image.size)
        rounded.paste(image, (0, 0), mask=mask)
        return rounded

    def image(self, chart, displayLegend=True):
        if displayLegend:
            main = Image.open("index/assets/predict.png").convert("RGBA")
        else:
            main = Image.open("index/assets/chart.png").convert("RGBA")
        legend = Image.open("index/assets/legend.png").convert("RGBA")
        chartImg = Image.open(chart).resize((2400, 1200)).convert("RGBA")

        img = Image.new(mode="RGB", size=(2500, 1500), color=(10, 19, 27))
        # serverIcon may be a URL (string), a local path, or a file-like object.
        try:
            if isinstance(self.serverIcon, str) and self.serverIcon.startswith("http"):
                resp = requests.get(self.serverIcon, timeout=5)
                resp.raise_for_status()
                serverIcon = Image.open(io.BytesIO(resp.content)).convert("RGBA").resize((93, 93))
            elif hasattr(self.serverIcon, "read"):
                # file-like object (BytesIO etc.)
                try:
                    self.serverIcon.seek(0)
                except Exception:
                    pass
                serverIcon = Image.open(self.serverIcon).convert("RGBA").resize((93, 93))
            else:
                serverIcon = Image.open(self.serverIcon).convert("RGBA").resize((93, 93))
        except Exception:
            # fallback to bundled placeholder icon
            try:
                serverIcon = Image.open("index/assets/placeholderIcon.jpg").convert("RGBA").resize((93, 93))
            except Exception:
                # last-resort: create a blank icon
                serverIcon = Image.new("RGBA", (93, 93), (112, 128, 144, 255))

        #Compositing
        img.paste(chartImg, (50, 250), mask=chartImg)
        img.paste(serverIcon, (1045, 76), serverIcon)
        if displayLegend:
            blur = chartImg.crop(box=(18, 18, 150, 242)).filter(ImageFilter.GaussianBlur(8))
            blurred = self._rounded(blur, 24)
            img.paste(blurred, (68, 269), mask=blurred)
            img.paste(legend, (24, 224), legend)
        img.paste(main, (0, 0), main)

        canvas = ImageDraw.Draw(im=img)
        canvas.text(xy=(1153, 75), text=self.serverName, font=self._font(48), fill="white")
        canvas.text(xy=(1153, 135), text=self.serverInvite.replace("https://", ""), font=self._font(28), fill=(112, 128, 144))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

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

                config = ph(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True, n_changepoints=15, changepoint_prior_scale=0.05, changepoint_range=0.8)
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
        window = history[(history.index >= today - timedelta(days=365)) & (history.index <= today)]

        curPrice = window["Close"].iloc[-1]
        lastDate = window.index[-1]
        
        points = []
        prophetTrend = self._prophetBacktest(history=window, lastDate=lastDate, curPrice=curPrice, histories=weights, forward=1)
        if prophetTrend is None: raise ValueError("Prophet generation failed")
        points = prophetTrend
        return points[0][1]