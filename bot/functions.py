from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import math
import threading
import time
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime, timedelta
from prophet import Prophet as ph
from pyfonts import set_default_font, load_google_font
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import themes
import requests
import psycopg2 as pg
import os

#Setup
matplotlib.use("Agg")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
set_default_font(load_google_font("Montserrat", weight="bold"))

connection = pg.connect(dbname="QuantumBot",user=os.getenv("PG_USERNAME"),password=os.getenv("PG_PASSWORD"),host="localhost")
if not connection: raise Exception("Cannot connect to database")

class Stamp:
    def __init__(self, name, url, icon, styles, factors=[]):
        self.serverName = name
        self.serverInvite = str(url)
        self.serverIcon = icon
        self.factors = factors
        self.styles:str = styles

    def _font(self, size: int): return ImageFont.truetype(font="bot/assets/Montserrat-Bold.ttf", size=size)

    def _rounded(self, image: Image.Image, radius: int) -> Image.Image:
        image = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        rounded = Image.new("RGBA", image.size)
        rounded.paste(image, (0, 0), mask=mask)
        return rounded

    def image(self, chart, displayLegend=True):
        if displayLegend: main = Image.open("bot/assets/template.png").convert("RGBA")
        else: main = Image.open("bot/assets/template.png").convert("RGBA")
        legend = Image.open("bot/assets/legend.png").convert("RGBA")
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
                try: self.serverIcon.seek(0)
                except Exception: pass
                serverIcon = Image.open(self.serverIcon).convert("RGBA").resize((93, 93))
            else: serverIcon = Image.open(self.serverIcon).convert("RGBA").resize((93, 93))
        except Exception:
            # fallback to bundled placeholder icon
            try: serverIcon = Image.open("bot/assets/placeholderIcon.jpg").convert("RGBA").resize((93, 93))
            except Exception: serverIcon = Image.new("RGBA", (93, 93), (112, 128, 144, 255)) # last-resort: create a blank icon

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
        canvas.text(xy=(688,95) if "predict" in self.styles else (709,95),text=self.styles, font=self._font(48), fill=themes.brand)
        canvas.text(xy=(2430,270), text="Source: finance.yahoo.com", font=self._font(15), fill=(56,68,80), align="right", anchor="rt")
        canvas.text(xy=(2430,290), text="Valid as of: "+datetime.now().strftime("%m/%d/%Y @ %H:%M:%S"), font=self._font(15), fill=(56,68,80), align="right", anchor="rt")

        bbox = [(1745,84),(2441,184)]
        width = bbox[1][0]-bbox[0][0]

        if "predict" in self.styles:
            width = math.floor(width/2)
            bbox1 = [(1750,84),(2441-width,184)]
            bbox2 = [(1740+width,84),(2441,184)]
            canvas.rectangle(bbox1, fill="black") #test bounding boxes
            canvas.rectangle(bbox2, fill="blue")
            canvas.text(xy=(1953, 58), text="Considerations Affecting Prediction:", font=self._font(16), fill=(112, 128, 144))
        else:
            width = math.floor(width/3)
            bbox1 = [(1750,84),(2441-width*2,184)]
            bbox2 = [(1740+width+20,84),(2441-width,184)]
            bbox3 = [(1740+width*2,84),(2441,184)]

            """canvas.rectangle(bbox1, fill="black") #test bounding boxes
            canvas.rectangle(bbox2, fill="blue")
            canvas.rectangle(bbox3, fill="red")"""

            canvas.text(xy=(2007, 58), text="Current Ticker Information:", font=self._font(16), fill=(112, 128, 144))
            if self.factors:
                # segment 1: 52wk, volume, mkt cap
                canvas.text(xy=bbox1[0], text="• 52 Week High: %s\n• 52 Week Low: %s\n• Volume: %s\n• Average Volume: %s\n• Market Cap: %s"%(round(self.factors["get52wkHigh"],2),round(self.factors["get52wkLow"],2),Humanizer.suffix(self.factors["getVolume"]),Humanizer.suffix(self.factors["getAvgVolume"]),Humanizer.suffix(self.factors["getMktCap"])), font=self._font(16), fill='white')
                # segment 2: p/e, eps, yield (a, m)
                canvas.text(xy=bbox2[0], text=( "• P/E Ratio: {}\n" "• EPS Ratio: {}\n" "• Beta: {}\n" "• Annual Yield: {}%\n" "• Monthly Yield: {}%" ).format( round(self.factors["getPERatio"], 2), round(self.factors["getEPSRatio"], 2), round(self.factors["getBeta"], 2), round(self.factors["getAnnualYield"], 2), round(self.factors["getMonthlyYield"], 2)), font=self._font(16), fill='white')
                # segment 3: 
                canvas.text(xy=bbox3[0], text = ( f"• Div. Amount: {self.factors['getDividendAmount']}\n" f"• Div. Change: {self.factors['getDividendChange']}\n" f"• Ex. Div. Date: {self.factors['getExDividendDate']}\n" f"• Pay Date: {self.factors['getPayDate']}" ), font=self._font(16), fill='white')

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
        self._capacity = 64 # max cached items
        self._inflections = 20 # number of bends
        self._flexibility = 0.05 # controls over/underfitting
        self._range = 0.8 # up to what percentage of the history prophet learns from (0.0-1.0)
        self._samples = 1500 # how smooth, more = smoother
        self._seasonality = 10 # controls over/underfitting of the seasons
    
    def _forecast(self, stock, history, configs, today, forward=90):
        today = datetime.strptime(today, "%Y-%m-%d") if isinstance(today, str) else today
        if today not in history.index:
            locs = history.index.get_indexer([today], method='pad')
            if locs[0] == -1: return None
            lastDate = history.index[locs[0]]
        else: lastDate = today

        earnings = None
        dates = stock.get_earnings_dates()
        if dates is not None and not dates.empty:
            earnings = pd.DataFrame({
                'holiday': 'earnings',
                'ds': dates.index.tz_localize(None), # Remove timezone
                'lower_window': 0,
                'upper_window': 1, # Impact usually felt day-of and day-after
            })

        curPrice = history.loc[lastDate]["Close"]
        results = []

        for h, settings in configs.items():
            startDate = lastDate - timedelta(days=int(h))
            window = history[(history.index > startDate) & (history.index <= lastDate)]
            
            # Minimum data check
            if len(window) < 50:
                results.append(np.full(forward, curPrice))
                continue
            
            key = (lastDate.isoformat(), h, tuple(window["Close"].values[-5:]), "LOGISTIC_V2") # check cache (WITH check to make sure cache is on version 2 running prophet logistic config)
            with self._thread: cached = self._cache.get(key)
            
            if cached is not None:
                results.append(cached)
                continue

            data = window.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
            data["ds"] = data["ds"].dt.tz_localize(None)

            limit = 0.3 #+-30% MAX
            cap = max(data['y'].max(), curPrice*(1+limit))
            floor = min(data['y'].min(), curPrice*(1-limit))

            data['cap'] = cap
            data['floor'] = floor

            # 5. Prophet Configuration
            config = ph(
                growth='logistic', # new system has cap and floor instead of linear infinite approximation
                holidays=earnings,
                daily_seasonality=False, 
                yearly_seasonality=True, 
                weekly_seasonality=True, 
                seasonality_prior_scale=self._seasonality,
                n_changepoints=self._inflections, #reduction for overfitting
                changepoint_prior_scale=self._flexibility, # how stiff/elastic trend is
                changepoint_range=self._range,
                #uncertainty_samples=self._samples,
                uncertainty_samples=0, # speed up calc
            )
            
            try:
                # Fit/predict can produce numerical overflow warnings (logistic growth exp); suppress them and sanitize outputs
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    config.fit(data)
                    future = config.make_future_dataframe(periods=forward, freq=settings[1]) 
                    future['cap'] = cap
                    future['floor'] = floor
                    fcst = config.predict(future)

                rawTrend = fcst.tail(forward)["yhat"].values
                rawTrend = np.nan_to_num(rawTrend, nan=curPrice, posinf=curPrice, neginf=curPrice)

                # attach to current day value (offset)
                if len(rawTrend) > 0: curve = rawTrend + (curPrice - rawTrend[0]) #curve = np.clip(curve, floor, cap) #clip incase past limits
                else: curve = np.full(forward, curPrice)

                if not np.all(np.isfinite(curve)): curve = np.full(forward, curPrice)
                with self._thread: self._cache[key] = curve
                results.append(curve)
            except Exception: results.append(np.full(forward, curPrice))

        return np.vstack(results)

    def _smapeLoss(self, w, raw, actuals):
        tune = 0

        preds = np.dot(w, raw)
        targets = actuals
        denom = (np.abs(targets) + np.abs(preds))
        diff = 2 * np.abs(preds - targets) / (denom + 1e-8)
        smape = np.mean(diff)

        pStart = targets[0]
        pEnd = preds[-1]
        change = abs((pEnd-pStart)/pStart)
        penalty = 0
        #penalty = tune*np.sum(w*np.log((w+1e-8)*5)) #normalization
        if change > 0.30: penalty = (change - 0.30) * 2.0
        
        return smape + penalty

    def clean(self, values): return self.clean(values[0]) if len(values) < 2 else values

    def _liveTrain(self, ticker, progress=None):
        ticker = str(ticker).upper()

        end = (datetime.now() - timedelta(days=30)).replace(tzinfo=None)
        start = (end - timedelta(days=365*5)).replace(tzinfo=None)

        cursor = connection.cursor()
        if not cursor: raise Exception("ERROR: Failed to create cursor")
        # parameterized select to check if ticker already exists
        cursor.execute("SELECT weight FROM ticker WHERE ticker = %s;", (ticker,))
        row = cursor.fetchone()

        default_weight = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
        # If there's an existing row, use it as starting weight and mark as update; otherwise create defaults
        if row is None: weight = default_weight
        else:
            try: weight = self.clean(row[0])
            except Exception: weight = default_weight

        bestWeight = weight[0]

        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sectorKey", info.get("quoteType", "uncategorized")).lower()
        ind = yf.Industry(info.get("industryKey")).name.lower() if info.get("industryKey") else str.lower(info.get("category")) if info.get("category") else "unknown"
        history = stock.history(start=start-timedelta(days=730), end=end, interval="1d") # 2018 to give prophet data to base off of
        if history.empty: return
        
        # Remove timezone from index to allow naive datetime string slicing
        if history.index.tz is not None: history.index = history.index.tz_localize(None)
        window = history.loc[start.strftime('%Y-%m-%d'):end.strftime('%Y-%m-%d')]
        daily = window.resample("D").interpolate()
        if daily.index.tz is not None: daily.index = daily.index.tz_convert("America/New_York").tz_localize(None)
        origins = window["Close"].resample("2W-FRI").last().dropna()

        bias = None
        weights = None
        errors = []
        for origin, price in origins.items(): #origin = fridays
            if progress: progress(f"Training: {origin} (this may take a few minutes)")
            bias = {90:[bestWeight[0], "ME"], 180:[bestWeight[1], "ME"], 365:[bestWeight[2], "D"], 730:[bestWeight[3], "W"], 1825:[bestWeight[4], "YS"]}
            rawCurves = self._forecast(stock, window, bias, origin, forward=90)
            
            if rawCurves is None: continue
            targetDates = [origin + timedelta(days=i) for i in range(90)]
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

            const = ({'type': 'eq', 'fun': lambda w:  np.sum(w) - 1.0})
            bounds = ((0.0,1.0),(0.0,1.0),(0.0,1.0),(0.05,1.0),(0.05,1.0))
            initGuess = np.array(bestWeight, dtype=float)
            initGuess = initGuess / np.sum(initGuess)
            
            #res = minimize(self._smapeLoss, initGuess, args=(matrix, targets), method='SLSQP', bounds=bounds, constraints=const)
            res = minimize(self._smapeLoss, initGuess, args=(matrix, targets), method='SLSQP')
            bestWeight = res.x.tolist()
            bestError = res.fun
            try: errors.append(float(res.fun))
            except Exception: pass
            
            prevInd, countInd = weight
            adjustment = 0.05 #equal
            avgInd = [prevInd[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevInd))] #ema
            weights = [avgInd,countInd+1]
            #print(origin.date(), bestError, str(round(adjustment*100,2))+"%", bestWeight)
        timestamp = str(math.floor(int(datetime.now().timestamp())))
        if weights is None: weights = weight

        def _serialize(o):
            if isinstance(o, np.generic): return o.item() # numpy scalar -> native python scalar
            if isinstance(o, (np.ndarray,)): return o.tolist() # numpy arrays -> lists
            raise TypeError(f"Type {type(o)} not JSON serializable")

        serialized = json.dumps(weights, default=_serialize)

        # compute average error across optimization runs and use that as accuracy
        avgError = float(sum(errors) / len(errors)) if len(errors) > 0 else (float(bestError) if 'bestError' in locals() and bestError is not None else 0.0)
        acc = avgError

        cursor.execute(
            """
            INSERT INTO ticker (ticker, sector, industry, accuracy, weight, updated)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                accuracy = EXCLUDED.accuracy,
                weight = EXCLUDED.weight,
                updated = EXCLUDED.updated;
            """,
            (ticker, sector, ind, acc, serialized, timestamp))
        connection.commit()
        cursor.close()
        return weights

    def project(self, ticker, model, serverName, serverInvite, serverIcon, progress=None):
        forward = 90
        ticker = str(ticker).upper()
        stock = yf.Ticker(ticker)
        history = stock.history(period="5y", interval="1d") if model != 0 else stock.history(period="1wk") # use 5y for extrapolation, use 1wk for implied (only needed for prev values to display) 
        if history.empty: return None
        
        curPrice = history["Close"].iloc[-1]
        lastDate = history.index[-1]
        plotHistory = history[history.index > lastDate - timedelta(days=14)]
        
        quantiles = np.linspace(0.05, 0.95, 11)
        futureDays = np.arange(0, forward + 1) # 0 to 90 (91 points)

        cursor = connection.cursor()
        if progress: progress("Retrieving Weights...")
        cursor.execute(f"select weight, updated from ticker where ticker = '{ticker}'")
        rows = cursor.fetchone()

        points = []
        if model != 1:
            ivPoints = self._impliedVolatility(stock, lastDate, forward, curPrice, quantiles, futureDays)
            points = ivPoints if ivPoints is not None else []
        
        # skip backtest for IV only
        if model != 0: # only train if necessary
            bias = None
            train = True
            if rows is not None:
                if len(json.dumps(rows)) > 1:
                    rows = self.clean(rows)
                    weight:list = rows[0]
                    updated = time.time()-int(rows[1])
                    if weight:
                        if updated < 432000: # 432000 = 5d in s
                            bias = weight
                            train = False
            if train:
                if progress: progress("Retraining Weights... (this may take a few minutes)")
                bias = self._liveTrain(ticker=ticker, progress=progress)
            if bias is None or (not hasattr(bias, "__getitem")) or len(bias) == 0: bias = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]
            bias = bias[0]

            histories = {90: [bias[0], "ME"], 180: [bias[1], "ME"], 365: [bias[2], "D"], 730: [bias[3], "W"], 1825: [bias[4], "YS"]} #fallbacks
            
            # live optimization mini backtest on the spot: 
            startDate = lastDate - timedelta(days=90)
            window = history[history.index <= startDate]
            actuals = (history[(history.index > startDate) & (history.index <= lastDate)]["Close"].values)[:forward] # truncate just in case
            
            if len(actuals) > 20:
                raw = self._forecast(stock, window, histories, startDate, forward=len(actuals))
                const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}) #constraints
                bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.05, 1.0), (0.05, 1.0)) # give weight to long term memory to 2y + 5y
                
                guess = [0.2, 0.2, 0.2, 0.2, 0.2]
                result = minimize(self._smapeLoss, guess, args=(raw, actuals), method='SLSQP', bounds=bounds, constraints=const)
                bestWeight = result.x
            else: bestWeight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            if progress: progress("Generating Shareable Image...")

            future = self._forecast(stock, history, histories, lastDate, forward=forward+1)
            if future is None: return None
            prophetTrend = np.dot(bestWeight, future)
            prophetSigma = 3
            
            if model == 1:
                if prophetTrend is None: raise ValueError("Prophet generation failed")
                points = np.array([prophetTrend + (norm.ppf(q) * prophetSigma) for q in quantiles])
            elif model == 2 and len(points) > 0 and prophetTrend is not None:
                spread = points - curPrice
                points = np.array([prophetTrend + spread[i] for i in range(len(quantiles))])

        if len(points) == 0: return None
        points = np.maximum(points, 0.01)
        futureDates = [lastDate + timedelta(days=int(d)) for d in futureDays]
        
        fig, ax = self._setupFigure()
        ax.plot(plotHistory.index, plotHistory["Close"], color=themes.brand, linewidth=2, zorder=10)
        
        minY = min(plotHistory["Close"].min(), np.min(points))
        maxY = max(plotHistory["Close"].max(), np.max(points))
        
        self._drawGradient(ax, mdates.date2num(plotHistory.index), plotHistory["Close"].values, minY, themes.brand)
        
        mid = len(quantiles) // 2
        median = points[mid]

        for i in range(mid): ax.fill_between(futureDates, points[i], points[-(i+1)], color=themes.brand, alpha=0.15, lw=0)
        ax.plot(futureDates, median, color=themes.brand, linewidth=2, linestyle=("dashed" if model != 0 else "solid"))

        allDates = list(plotHistory.index) + futureDates
        self._formatAxes(ax, allDates, minY, maxY, median[-1], formatX=True)
        
        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax.annotate(f"${median[-1]:.2f}", xy=(1, median[-1]), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)
        
        plt.title(f"{str.upper(ticker)} Prediction (90d)", fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center")

        chartBuf = self._buffer(fig)
        if progress: progress("Finalizing Image...")
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon, styles="/predict").image(chartBuf)
    
    def _impliedVolatility(self, stock, lastDate, forward, curPrice, quantiles, futureDays):
        anchorsY = [[curPrice] * len(quantiles)] 
        anchorsX = [0]
        
        options = stock.options
        if len(options) <= 1: return None

        start = lastDate.date()

        for exp in options:
            try:
                expDate = datetime.strptime(exp, "%Y-%m-%d").date()
                daysDiff = (expDate - start).days
                if daysDiff < 0: continue
                
                if daysDiff > forward + 15: break
                
                opt = stock.option_chain(exp)
                
                centerStrike = curPrice
                calls = opt.calls.iloc[(opt.calls["strike"] - centerStrike).abs().argsort()[:2]]
                puts = opt.puts.iloc[(opt.puts["strike"] - centerStrike).abs().argsort()[:2]]
                
                valid_ivs = pd.concat([calls["impliedVolatility"], puts["impliedVolatility"]])
                valid_ivs = valid_ivs[valid_ivs > 0.001] # filter 0 or ~0
                
                if valid_ivs.empty: continue
                meanIV = valid_ivs.mean()

                # even if daysDiff is 0 or 1, we force tYears to be at least 1/365; this prevents the square root of time from becoming 0 and collapsing the graph
                effective_days = max(daysDiff, 1.0)
                tYears = effective_days / 365.0
                
                expPrices = []
                for q in quantiles:
                    z = norm.ppf(q)
                    # Geometric Brownian Motion
                    projection = curPrice*np.exp(-1*meanIV**2*tYears+meanIV*np.sqrt(tYears)*z)+0.04
                    expPrices.append(projection)
                
                anchorsX.append(max(daysDiff, 1))
                anchorsY.append(expPrices)
            except Exception:
                continue

        if len(anchorsX) < 2:
            anchorsX.append(forward)
            anchorsY.append([curPrice] * len(quantiles))

        yTransposed = np.array(anchorsY).T 
        points = []
        for series in yTransposed:
            cs = CubicSpline(anchorsX, series, bc_type="natural")
            points.append(cs(futureDays))
        return np.array(points)

    def _setupFigure(self):
        plt.rc("font", size=10)
        fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
        fig.patch.set_facecolor(color=themes.bgDark)
        ax.set_facecolor(themes.bgDark)
        return fig, ax
    
    def _formatAxes(self, ax, dates, minY, maxY, lastPrice=None, formatX=True):
        if formatX:
            span = dates[-1] - dates[0]
            if span.days > 730: fmt = "%Y"
            elif dates[-1].year != dates[0].year: fmt = "%b %Y"
            else: fmt = "%b %d"
            
            ax.xaxis.set_major_locator(MaxNLocator(nbins=24, min_n_ticks=16))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.tick_params(axis="x", rotation=90, colors=themes.grayDark, labelcolor=themes.grayDark)
        
        yRange = maxY - minY
        if yRange == 0: yRange = 1
        rawStep = yRange / 20
        allowedSteps = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        step = min(allowedSteps, key=lambda x: abs(x - rawStep))
        
        if lastPrice:
            ticksUp = np.arange(lastPrice, maxY * 1.05, step)
            ticksDown = np.arange(lastPrice - step, minY * 0.95, -step)
            customTicks = np.sort(np.concatenate((ticksDown, ticksUp)))
        else: customTicks = np.arange(minY, maxY, step)

        ax.set_yticks(customTicks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
        
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", colors=themes.grayDark, labelcolor=themes.grayDark)
        
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_color(themes.grayDark)
        ax.spines["bottom"].set_color(themes.grayDark)
        
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        if formatX:ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)
        ax.set_ylim(minY * 0.98, maxY * 1.02)
        if formatX: ax.set_xlim(dates[0], dates[-1])

    def _drawGradient(self, ax, xNums, yVals, minY, color):
        yFloor = minY * 0.90
        verts = [(xNums[0], yFloor)] + list(zip(xNums, yVals)) + [(xNums[-1], yFloor)]
        poly = Polygon(verts, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(poly)
        
        cTop = to_rgba(color, alpha=0.3)
        cBot = to_rgba(color, alpha=0.0)
        cmap = LinearSegmentedColormap.from_list("grad", [cBot, cTop])
        
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        im = ax.imshow(grad, aspect="auto", cmap=cmap, origin="lower", 
                       extent=[xNums[0], xNums[-1], yFloor, max(yVals)], zorder=1)
        im.set_clip_path(poly)

    def _buffer(self, fig):
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf

    def history(self, ticker, duration, interval, serverName, serverInvite, serverIcon, staticQuote, progress=None):
        stock = yf.Ticker(ticker)
        periods = ["1d","5d","1mo","3mo","6mo","1y","ytd","2y","5y","10y","max"]
        intervals = ["2m","15m","30m","60m","1d","5d","1mo","3mo"]

        preview = duration
        if duration == "ytd" or duration == "1y": # swap ytd and 1y because it's mixed up in api???? monitor the api for changes (prob a bug)
            swaps = ["ytd","1y"]
            duration = swaps[not bool(swaps.index(duration))]

        if progress: progress("Retrieving Historical Data...")
        if interval is None:
            interval = "1d"
            if duration in ["1d"]: interval = "2m"
            elif duration in ["5d"]: interval = "60m"
            elif duration in ["1mo", "3mo"]: interval = "1d"
            elif duration in ["6mo", "1y", "ytd"]: interval = "5d"
            elif duration in ["2y", "5y"]: interval = "1mo"
            else: interval = "3mo"
        else: assert interval in intervals, "Not valid interval"
        assert intervals.index(interval)-4 < periods.index(duration), "Interval more than period"

        def formatDate(x, pos=None):
            idx = int(x)
            if 0 <= idx < len(history):
                date = history.index[idx]
                string = ""
                if periods.index(duration) > periods.index("1mo") and periods.index(duration) <= periods.index("2y"): string+="%b " # if more than month and less than or year, append short month (Feb)
                if periods.index(duration) >= periods.index("5d") and periods.index(duration) <= periods.index("1mo"): string+="%a " # if 5d or same month, append short weekday (Mon)
                if periods.index(duration) >= periods.index("1mo") and periods.index(duration) <= periods.index("ytd") : string+="%d " # if more than week and less than a year, append day instead (01)
                if periods.index(duration) > periods.index("1y") : string+="%Y " # if more than year, also append year (2020)
                if intervals.index(interval) <= intervals.index("1d"): string+=(str(date.strftime("%I")).replace("0","")+":%M %p") # if interval less than a day, append time as AM/PM (09:00 PM)
                return date.strftime(string)
            return ""

        history = stock.history(period=duration, interval=interval)
        if history.empty: return None

        if history.index.tz is None: history.index = history.index.tz_localize("UTC")
        history.index = history.index.tz_convert("America/New_York")

        #plt.rc("font", size=10)
        fig = plt.figure(figsize=(20, 10), dpi=100)
        
        # Create a GridSpec with two rows, height ratio 3:1 (price:volume), no initial vertical space (hspace=0.0)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0) 
        ax1 = fig.add_subplot(gs[0]) # Price chart (top)
        ax2 = fig.add_subplot(gs[1], sharex=ax1) # Volume chart (bottom), sharing x-axis
        
        fig.patch.set_facecolor(color=themes.bgDark)
        ax1.set_facecolor(themes.bgDark)
        ax2.set_facecolor(themes.bgDark)

        history = history.copy()
        history["x_index"] = np.arange(len(history))
        
        up = history[history.Close >= history.Open]
        down = history[history.Close < history.Open]
        width = 0.6
        width2 = 0.08 

        maxVol = history.Volume.max()
        ax2.set_ylim(0, maxVol) 
        
        vol_colors = [themes.brand if c >= o else themes.red for c, o in zip(history.Close, history.Open)]
        ax2.bar(history["x_index"], history.Volume, width=width, color=vol_colors, alpha=0.5)
        
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax1.spines["bottom"].set_color(themes.grayDark)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_color(themes.grayDark) # Set color for the bottom spine
        ax2.spines["right"].set_color(themes.grayDark)
        ax2.tick_params(axis="y", colors=themes.grayDark, labelcolor=themes.grayDark, labelsize=8)
        
        def volume(x, pos): return Humanizer.suffix(x)
        ax2.yaxis.set_major_formatter(FuncFormatter(volume))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))

        ax1.set_zorder(10)
        ax1.patch.set_visible(False)

        ax1.bar(up["x_index"], up.Close - up.Open, bottom=up.Open, width=width, color=themes.brand)
        ax1.bar(up["x_index"], up.High - up.Close, bottom=up.Close, width=width2, color=themes.brand)
        ax1.bar(up["x_index"], up.Low - up.Open, bottom=up.Open, width=width2, color=themes.brand)
        
        downColor = themes.red
        ax1.bar(down["x_index"], down.Close - down.Open, bottom=down.Open, width=width, color=downColor)
        ax1.bar(down["x_index"], down.High - down.Open, bottom=down.Open, width=width2, color=downColor)
        ax1.bar(down["x_index"], down.Low - down.Close, bottom=down.Close, width=width2, color=downColor)

        minY = history["Low"].min()
        maxY = history["High"].max()
        lastPrice = history["Close"].iloc[-1]

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.tick_params(axis='x', which='both', length=0)
        
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=min(64, len(history)), min_n_ticks=1, integer=True))
        ax2.xaxis.set_major_formatter(FuncFormatter(formatDate))
        uniques = sorted(set(ax2.get_xticks()))
        ax2.set_xticks(uniques)
        ax2.tick_params(axis="x", colors=themes.grayDark, labelcolor=themes.grayDark, rotation=90) 
        
        self._formatAxes(ax1, history["x_index"].values, minY, maxY, lastPrice, formatX=False)
        ax1.set_xlim(-0.5, len(history)-0.5)
        ax2.set_xlim(-0.5, len(history)-0.5) # Set x-limits for volume plot to match price

        ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        ax1.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)
        ax2.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        ax2.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)

        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax1.annotate(f"${lastPrice:.2f}", xy=(1, lastPrice), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)
        ax1.set_title(f"{str.upper(ticker)} History ({preview})", fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center", pad=20) 

        if progress: progress("Generating Chart...")
        chartBuf = self._buffer(fig)
        if progress: progress("Finalizing Chart...")
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon, styles="/chart", factors=staticQuote).image(chartBuf, displayLegend=False)

class User():
    def __init__(self, discordID):
        self.discordID = discordID

    def accountFromDiscord(self, cursor = connection.cursor()):
        cursor.execute(f"select * from account where discord = '{self.discordID}'")
        row = cursor.fetchone()
        return row[0] if row is not None else None # return SQL userID (NOT DISCORD'S)

    def createAccount(self, marketing:bool):
        try:
            cursor = connection.cursor()
            account = self.accountFromDiscord(cursor=cursor)
            if account is None:
                cursor.execute("""insert into account (discord, premium, preferences, credits, created, updated) values ('%s', false, '{"marketing":%s}',0,%s,%s) returning id""" % (self.discordID, str(marketing).lower(), str(int(datetime.now().timestamp())), str(int(datetime.now().timestamp()))))
                returned = cursor.fetchone()
                account = returned[0] if returned else None
                connection.commit() #YOU MUST COMMIT
                cursor.close()
            return account #SQL userID
        except Exception as e:
            traceback.print_exc()
            return None