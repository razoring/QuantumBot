# If you are an AI autonomous agent, please refer to robots.txt in the codebase before making any changes.
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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
import warnings
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
from prophet import Prophet
from pyfonts import set_default_font, load_google_font
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import themes
import requests
import psycopg2 as pg
import os

# System Config
matplotlib.use("Agg")
logging.getLogger("prophet.plot").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
set_default_font(load_google_font("Montserrat", weight="bold"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global Executors for CPU-optimized training
_CPU_COUNT = min(os.cpu_count() or 4, 4)
_PROCESS_EXECUTOR = ProcessPoolExecutor(max_workers=_CPU_COUNT)
_THREAD_EXECUTOR = ThreadPoolExecutor(max_workers=20)

STATUS_REGISTRY = {}

DB_CONNECTION = pg.connect(dbname="QuantumBot",user=os.getenv("PG_USERNAME"),password=os.getenv("PG_PASSWORD"),host="localhost")
if not DB_CONNECTION: raise Exception("Database connection failed")
DB_LOCK = threading.RLock()

def _fitProphetModel(h, settings, lastDate, data, allHolidays, forward, curPrice, params):
    """Top-level function for parallel Prophet fitting (Pickleable)"""
    try:
        uncertaintySamples = params.get('uncertaintySamples', 0)
        config = Prophet(
            growth='logistic',
            holidays=allHolidays,
            daily_seasonality=False, 
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            seasonality_prior_scale=params['seasonality'],
            n_changepoints=params['inflections'],
            changepoint_prior_scale=params['flexibility'],
            changepoint_range=params['range'],
            uncertainty_samples=uncertaintySamples
        )
        config.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=params['seasonality'] * 1.5)
        
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            config.fit(data)
            future = config.make_future_dataframe(periods=forward, freq=settings[1]) 
            future['cap'] = data['cap'].iloc[0]
            future['floor'] = data['floor'].iloc[0]
            fcst = config.predict(future)

        rawTrend = fcst.tail(forward)["yhat"].values
        rawTrend = np.nan_to_num(rawTrend, nan=curPrice, posinf=curPrice, neginf=curPrice)
        
        if config.uncertainty_samples > 0:
            try:
                samples = config.predictive_samples(future)
                rawSigma = np.std(samples['yhat'], axis=1)
                rawSigma = rawSigma[-forward:]
            except Exception:
                rawSigma = (fcst.tail(forward)["yhat_upper"].values - fcst.tail(forward)["yhat_lower"].values) / 2.56
        else:
            rawSigma = np.full(forward, curPrice * 0.02)

        rawSigma = np.nan_to_num(rawSigma, nan=curPrice*0.02, posinf=curPrice*0.02, neginf=curPrice*0.02)

        # Extract deltas and changepoints
        deltas = config.params['delta'].mean(axis=0)
        cp_dates = config.changepoints.values
        
        if len(rawTrend) > 0: curve = rawTrend + (curPrice - rawTrend[0])
        else: curve = np.full(forward, curPrice)
        
        if not np.all(np.isfinite(curve)): curve = np.full(forward, curPrice)
        return h, (curve, rawSigma, deltas, cp_dates)
    except Exception:
        return h, (np.full(forward, curPrice), np.full(forward, curPrice * 0.02), np.zeros(params['inflections']), np.array([]))

class Stamp:
    def __init__(self, name, url, icon, styles, factors):
        self._serverName = name
        self._serverInvite = str(url)
        self._serverIcon = icon
        self._factors = factors
        self._styles:str = styles

    def _font(self, size: int): return ImageFont.truetype(font="bot/assets/Montserrat-Bold.ttf", size=size)

    def _rounded(self, image: Image.Image, radius: int) -> Image.Image:
        image = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        rounded = Image.new("RGBA", image.size)
        rounded.paste(image, (0, 0), mask=mask)
        return rounded

    def image(self, chartPath, displayLegend=True):
        mainTemplate = Image.open("bot/assets/generation/template.png").convert("RGBA")
        legendOverlay = Image.open("bot/assets/generation/legend.png").convert("RGBA")
        chartImg = Image.open(chartPath).resize((2400, 1200)).convert("RGBA")

        finalImg = Image.new(mode="RGB", size=(2500, 1500), color=(10, 19, 27))
        try:
            if isinstance(self._serverIcon, str) and self._serverIcon.startswith("http"):
                response = requests.get(self._serverIcon, timeout=5)
                response.raise_for_status()
                iconImg = Image.open(io.BytesIO(response.content)).convert("RGBA").resize((93, 93))
            elif hasattr(self._serverIcon, "read"):
                try: self._serverIcon.seek(0)
                except Exception: pass
                iconImg = Image.open(self._serverIcon).convert("RGBA").resize((93, 93))
            else: iconImg = Image.open(self._serverIcon).convert("RGBA").resize((93, 93))
        except Exception:
            try: iconImg = Image.open("bot/assets/icons/discord.jpg").convert("RGBA").resize((93, 93))
            except Exception: iconImg = Image.new("RGBA", (93, 93), (112, 128, 144, 255))

        finalImg.paste(chartImg, (50, 250), mask=chartImg)
        finalImg.paste(iconImg, (1045, 76), iconImg)
        if displayLegend:
            blurZone = chartImg.crop(box=(18, 18, 150, 242)).filter(ImageFilter.GaussianBlur(8))
            blurredMask = self._rounded(blurZone, 24)
            finalImg.paste(blurredMask, (68, 269), mask=blurredMask)
            finalImg.paste(legendOverlay, (24, 224), legendOverlay)
        finalImg.paste(mainTemplate, (0, 0), mainTemplate)

        draw = ImageDraw.Draw(im=finalImg)
        draw.text(xy=(1153, 75), text=self._serverName, font=self._font(48), fill="white")
        draw.text(xy=(1153, 135), text=self._serverInvite.replace("https://", ""), font=self._font(28), fill=(112, 128, 144))
        draw.text(xy=(688,95) if "predict" in self._styles else (709,95),text=self._styles, font=self._font(48), fill=themes.brand)
        draw.text(xy=(2430,270), text="Source: finance.yahoo.com", font=self._font(15), fill=(56,68,80), align="right", anchor="rt")
        draw.text(xy=(2430,290), text="Valid as of: "+datetime.now().strftime("%m/%d/%Y @ %H:%M:%S"), font=self._font(15), fill=(56,68,80), align="right", anchor="rt")

        contentBBox = [(1745,84),(2441,184)]
        contentWidth = contentBBox[1][0]-contentBBox[0][0]

        if "predict" in self._styles:
            draw.text(xy=(1953, 58), text="Considerations Affecting Prediction:", font=self._font(16), fill=(112, 128, 144))
            
            sorted_factors = sorted(
                self._factors, 
                key=lambda x: x.get('impact', {}).get('val', 0) if isinstance(x, dict) else 0, 
                reverse=True
            )[:10]

            # Dynamic column split with padding
            leftStartX = 1750
            maxLeftWidth = 0
            maxRightWidth = 0
            rightPadding = 20 # Padding from the right edge of contentBBox
            
            for i, factor in enumerate(sorted_factors):
                if isinstance(factor, dict) and "impact" in factor:
                    fullStr = f"{factor['impact']['symbol']} {factor['impact']['pct']} {factor['label']}"
                else: fullStr = str(factor)
                try: w = draw.textlength(fullStr, font=self._font(16))
                except AttributeError: w = self._font(16).getsize(fullStr)[0]
                
                if i < 5: maxLeftWidth = max(maxLeftWidth, w)
                else: maxRightWidth = max(maxRightWidth, w)
            
            # Initial right Start X based on left labels
            rightStartX = leftStartX + maxLeftWidth + 40
            
            # Ensure right column has padding on the right edge of contentBBox (2441)
            # If (rightStartX + maxRightWidth) > (2441 - rightPadding), we need to push back or cap
            if (rightStartX + maxRightWidth) > (2441 - rightPadding):
                # Attempt to nudge left if we have room in the center
                overage = (rightStartX + maxRightWidth) - (2441 - rightPadding)
                rightStartX = max(leftStartX + maxLeftWidth + 20, rightStartX - overage)

            for i, factor in enumerate(sorted_factors):
                posX = leftStartX if i < 5 else rightStartX
                posY = 84 + (i % 5) * 20
                
                if isinstance(factor, dict) and "impact" in factor:
                    impactStr = f"{factor['impact']['symbol']} {factor['impact']['pct']} "
                    draw.text((posX, posY), impactStr, font=self._font(16), fill=factor['impact']['color'])
                    
                    try: prefixWidth = draw.textlength(impactStr, font=self._font(16))
                    except AttributeError: prefixWidth = self._font(16).getsize(impactStr)[0]
                    draw.text((posX + prefixWidth, posY), factor['label'], font=self._font(16), fill='white')
                else:
                    draw.text((posX, posY), str(factor), font=self._font(16), fill='white')
        else:
            draw.text(xy=(2007, 58), text="Current Ticker Information:", font=self._font(16), fill=(112, 128, 144))
            if self._factors:
                groups = [
                    [
                        ("52 Week High", f'${round(self._factors["get52wkHigh"],2)}'),
                        ("52 Week Low", f'${round(self._factors["get52wkLow"],2)}'),
                        ("Volume", Humanizer.suffix(self._factors["getVolume"])),
                        ("Avg Volume", Humanizer.suffix(self._factors["getAvgVolume"])),
                        ("Market Cap", Humanizer.suffix(self._factors["getMktCap"]))
                    ],
                    [
                        ("P/E Ratio", round(self._factors["getPERatio"], 2)),
                        ("EPS Ratio", round(self._factors["getEPSRatio"], 2)),
                        ("Beta", round(self._factors["getBeta"], 2)),
                        ("Annual Yield", f'{round(self._factors["getAnnualYield"], 2)}%'),
                        ("Monthly Yield", f'{round(self._factors["getMonthlyYield"], 2)}%')
                    ],
                    [
                        ("Div. Amount", self._factors['getDividendAmount']),
                        ("Div. Change", self._factors['getDividendChange']),
                        ("Ex Div. Date", self._factors['getExDividendDate']),
                        ("Pay Date", self._factors['getPayDate'])
                    ]
                ]

                startX = 1750
                columnWidths = []
                for group in groups:
                    maxW = 0
                    for label, val in group:
                        fullStr = f"• {label}: {val}"
                        try: w = draw.textlength(fullStr, font=self._font(16))
                        except AttributeError: w = self._font(16).getsize(fullStr)[0]
                        maxW = max(maxW, w)
                    columnWidths.append(maxW + 40)

                currentX = startX
                for group_idx, group in enumerate(groups):
                    for row_idx, (label, val) in enumerate(group):
                        posY = 84 + row_idx * 20
                        labelText = f"• {label}: "
                        valText = str(val)
                        
                        draw.text((currentX, posY), labelText, font=self._font(16), fill='white')
                        try: lw = draw.textlength(labelText, font=self._font(16))
                        except AttributeError: lw = self._font(16).getsize(labelText)[0]
                        
                        draw.text((currentX + lw, posY), valText, font=self._font(16), fill=themes.brand)
                    
                    currentX += columnWidths[group_idx]

        buf = io.BytesIO()
        finalImg.save(buf, format="PNG")
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
        with ThreadPoolExecutor(max_workers=3) as executor:
            fFast = executor.submit(self._symbol.get_fast_info)
            fInfo = executor.submit(lambda: self._symbol.info)
            fCalendar = executor.submit(lambda: self._symbol.calendar)
            
            self._fastInfo = fFast.result()
            self._info = fInfo.result()
            self._calendar = fCalendar.result()
        self._cachedHistory = None 

    def _getHistory(self, period="1y"):
        if self._cachedHistory is None:
            self._cachedHistory = self._symbol.history(period=period)
        return self._cachedHistory

    def getStockInfo(self): return self._info
    def getFastInfo(self): return self._fastInfo
    def getCalendar(self): return self._calendar
    def getCurrentPrice(self): return self._fastInfo.get("lastPrice", 0)
    def getDayOpen(self): return self._fastInfo.get("open", 0)
    def getDayClose(self): return self._fastInfo.get("previousClose", 0)

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
        if "dividendYield" in self._info and self._info["dividendYield"] is not None: return round(self._info["dividendYield"] * 100, 2)
        rate = self._info.get("trailingAnnualDividendRate")
        price = self.getCurrentPrice()
        if rate and price: return round((rate / price) * 100, 2)
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
        
    def getAnalystRating(self):
        return self._info.get("recommendationKey", "none")

class Charts:
    _TRAINING_REGISTRY = {}
    _REGISTRY_LOCK = threading.RLock()
    _CACHE = {}
    _CACHE_LOCK = threading.RLock()
    _CAPACITY = 64
    _SECTOR_MAP = {
        'technology': 'XLK',
        'healthcare': 'XLV',
        'financial-services': 'XLF',
        'energy': 'XLE',
        'industrials': 'XLI',
        'consumer-cyclical': 'XLY',
        'consumer-defensive': 'XLP',
        'utilities': 'XLU',
        'real-estate': 'XLRE',
        'basic-materials': 'XLB',
        'communication-services': 'XLC'
    }
    _MACRO_MAP = {
        'rates': '^TNX',
        'energy': 'CL=F',
        'economy': '^GSPC'
    }
    _MACRO_CORRELATIONS = {
        'rates': {
            'financial-services': 1,
            'technology': -1,
            'real-estate': -1,
            'utilities': -1,
            'healthcare': -0.5,
            'default': -1
        },
        'energy': {
            'energy': 1,
            'basic-materials': 0.5,
            'consumer-cyclical': -1,
            'industrials': -1,
            'default': -0.5
        },
        'economy': {
            'default': 1
        }
    }

    def evaluate(self, ticker: str):
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        
        # Get YTD data
        today = datetime.now()
        start_date = f"{today.year}-01-01"
        history = stock.history(start=start_date, interval="1d")
        
        # Ensure we have at least 90 days + some training context
        if len(history) < 120:
            history = stock.history(period="1y", interval="1d")
            
        if len(history) < 95:
            return None

        # Split: Fixed 90-day evaluation window
        split_idx = len(history) - 90
        train_df = history.iloc[:split_idx]
        test_df = history.iloc[split_idx:]
        
        # Get current model weights from DB
        currentWeights = [0.2] * 5
        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
                cursor.execute("SELECT weight FROM ticker WHERE ticker = %s", (ticker,))
                row = cursor.fetchone()
                if row and row[0]:
                    try:
                        decoded = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                        currentWeights = decoded[0]
                    except: pass

        # Fit and Forecast using the training set
        histories = {
            90: [currentWeights[0], "D"], 
            180: [currentWeights[1], "D"], 
            365: [currentWeights[2], "D"], 
            730: [currentWeights[3], "D"], 
            1825: [currentWeights[4], "D"]
        }
        
        forward = len(test_df)
        last_train_date = train_df.index[-1]
        
        results = self._forecast(stock, train_df, histories, last_train_date, forward=forward, parallel=True, uncertaintySamples=0)
        if results is None: return None
        curves, sigmas, _ = results
        
        prediction = np.dot(currentWeights, curves)
        
        # Track price difference for each day
        actual_vals = test_df['Close'].values
        price_diffs = []
        for p, a in zip(prediction, actual_vals):
            price_diffs.append(p - a)

        # Matplotlib Plot (Plain and simple)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot full history
        ax.plot(history.index, history['Close'], label='Actual Price', color='#2c3e50', linewidth=1.5)
        
        # Plot Prediction
        pred_dates = test_df.index
        ax.plot(pred_dates, prediction, label='90-Day Backtest', color='#e74c3c', linestyle='--', linewidth=2)
        
        # Vertical line for split
        ax.axvline(last_train_date, color='black', alpha=0.3, linestyle=':', label='Training End')
        
        # Labels and Title
        ax.set_title(f"Accuracy Evaluation: {ticker} (90-Day Backtest)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Timeline")
        ax.set_ylabel("Price (USD)")
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Error Calculation
        smape = np.mean(2 * np.abs(np.array(price_diffs)) / (np.abs(prediction) + np.abs(actual_vals))) * 100
        avg_diff = np.mean(np.abs(price_diffs))
        
        info_text = f"Backtest Window: 90 Days\nAvg Daily Error: ${avg_diff:.2f}\nSMAPE: {smape:.2f}%"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        test_range = (float(test_df['Close'].min()), float(test_df['Close'].max()))
        return buf, price_diffs, test_range


    def __init__(self):
        self._TTL = 60*60*24
        self._INFLECTIONS = 22
        self._FLEXIBILITY = 0.5
        self._RANGE = 0.9
        self._SAMPLES = 1500
        self._SEASONALITY = 0.05 #affects amplitude
        self._CONSTRAINTS = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1.0})
        self._BOUNDS = ((0.0,1.0),(0.0,1.0),(0.0,1.0),(0.05,1.0),(0.05,1.0))
    
    def _forecast(self, stock, history, configs, today, forward=90, parallel=True, uncertaintySamples=None):
        today = datetime.strptime(today, "%Y-%m-%d") if isinstance(today, str) else today
        if today not in history.index:
            locs = history.index.get_indexer([today], method='pad')
            if locs[0] == -1: return None
            lastDate = history.index[locs[0]]
        else: lastDate = today
        
        holidays = []

        try:
            dates = stock.get_earnings_dates()
            if dates is not None and not dates.empty:
                holidays.append(pd.DataFrame({
                    'holiday': 'earnings',
                    'ds': dates.index.tz_localize(None),
                    'lower_window': 0,
                    'upper_window': 1,
                }))
        except Exception: pass

        try:
            exTs = stock.info.get("exDividendDate")
            if exTs:
                holidays.append(pd.DataFrame({
                    'holiday': 'ex_dividend',
                    'ds': [pd.Timestamp(datetime.fromtimestamp(exTs).date())],
                    'lower_window': -1,
                    'upper_window': 0,
                }))
        except Exception: pass

        try:
            payDate = stock.calendar.get("Dividend Date")
            if payDate:
                holidays.append(pd.DataFrame({
                    'holiday': 'dividend_payout',
                    'ds': [pd.Timestamp(payDate)],
                    'lower_window': 0,
                    'upper_window': 0,
                }))
        except Exception: pass

        allHolidays = pd.concat(holidays) if holidays else None
        curPrice = history.loc[lastDate]["Close"]
        
        samples = uncertaintySamples if uncertaintySamples is not None else (min(self._SAMPLES, 500) if parallel else 0)
        prophetParams = {
            'seasonality': self._SEASONALITY,
            'inflections': self._INFLECTIONS,
            'flexibility': self._FLEXIBILITY,
            'range': self._RANGE,
            'uncertaintySamples': samples
        }

        tasks = []
        resultsMap = {}

        for h, settings in configs.items():
            startDate = lastDate - timedelta(days=int(h))
            window = history[(history.index > startDate) & (history.index <= lastDate)]
            window = window.resample("D").interpolate(method="linear").ffill().bfill()
            
            if len(window) < 50:
                resultsMap[h] = (np.full(forward, curPrice), np.full(forward, curPrice * 0.02), (np.zeros(prophetParams['inflections']), np.array([])))
                continue
            
            # Include weight values in cache key to ensure mutation changes the chart
            weightValues = tuple([settings[0] for settings in configs.values()])
            key = (lastDate.isoformat(), h, tuple(window["Close"].values[-5:]), weightValues, "LOGISTIC_V6")
            with self._CACHE_LOCK:
                cached = self._CACHE.get(key)
                if cached is not None:
                    timestamp, val = cached
                    if time.time() - timestamp < self._TTL:
                        # Ensure the cached value matches the new 3-tuple signature
                        if len(val) == 3:
                            resultsMap[h] = val
                            continue

            data = window.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
            data["ds"] = data["ds"].dt.tz_localize(None)
            limit = 0.3
            cap = max(data['y'].max(), curPrice*(1+limit))
            floor = min(data['y'].min(), curPrice*(1-limit))
            data['cap'] = cap
            data['floor'] = floor

            tasks.append((h, settings, lastDate, data, allHolidays, forward, curPrice, prophetParams, key))

        if tasks:
            if parallel:
                try:
                    futures = [_PROCESS_EXECUTOR.submit(_fitProphetModel, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]) for t in tasks]
                    for i, future in enumerate(as_completed(futures)):
                        hRes, (curve, sigma, deltas, cp_dates) = future.result()
                        resultsMap[hRes] = (curve, sigma, (deltas, cp_dates))
                        originalKey = next((t[8] for t in tasks if t[0] == hRes), None)
                        if originalKey:
                            with self._CACHE_LOCK:
                                self._CACHE[originalKey] = (time.time(), (curve, sigma, (deltas, cp_dates)))
                except Exception as e:
                    logging.error(f"ProcessPool Failure: {e}. Falling back to sequential processing.")
                    for t in tasks:
                        if t[0] not in resultsMap:
                            hRes, (curve, sigma, deltas, cp_dates) = _fitProphetModel(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
                            resultsMap[hRes] = (curve, sigma, (deltas, cp_dates))
                            with self._CACHE_LOCK:
                                self._CACHE[t[8]] = (time.time(), (curve, sigma, (deltas, cp_dates)))
            else:
                for t in tasks:
                    hRes, (curve, sigma, deltas, cp_dates) = _fitProphetModel(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
                    resultsMap[hRes] = (curve, sigma, (deltas, cp_dates))
                    with self._CACHE_LOCK:
                        self._CACHE[t[8]] = (time.time(), (curve, sigma, (deltas, cp_dates)))

        finalResults = [resultsMap[h] for h in configs.keys()]
        curves = np.vstack([r[0] for r in finalResults])
        sigmas = np.vstack([r[1] for r in finalResults])
        insights = [r[2] for r in finalResults]
        return (curves, sigmas, insights)

    def _smapeLoss(self, w, raw, actuals):
        preds = np.dot(w, raw)
        denom = (np.abs(actuals) + np.abs(preds))
        diff = 2 * np.abs(preds - actuals) / (denom + 1e-8)
        smape = np.mean(diff)

        pStart = actuals[0]
        change = abs((preds[-1]-pStart)/pStart)
        penalty = 0
        if change > 0.30: penalty = (change - 0.30) * 2.0
        return smape + penalty

    def clean(self, values): return self.clean(values[0]) if len(values) < 2 else values

    def _getIndustryAverageWeights(self, industry, sector):
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    # Try industry first
                    cursor.execute("SELECT weight FROM ticker WHERE industry = %s AND weight IS NOT NULL;", (industry,))
                    rows = cursor.fetchall()
                    
                    if len(rows) < 3 and sector:
                        # Try sector if industry data is sparse
                        cursor.execute("SELECT weight FROM ticker WHERE sector = %s AND weight IS NOT NULL;", (sector,))
                        rows = cursor.fetchall()
                    
                    if len(rows) >= 3:
                        allWeights = []
                        for row in rows:
                            try:
                                wData = row[0]
                                if isinstance(wData, str): wData = json.loads(wData)
                                if isinstance(wData, list) and len(wData) > 0:
                                    allWeights.append(wData[0])
                            except: continue
                        
                        if len(allWeights) >= 3:
                            return np.mean(allWeights, axis=0).tolist()
        except Exception: pass
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    def _liveTrain(self, ticker, userID=None):
        ticker = str(ticker).upper()
        now = datetime.now().replace(tzinfo=None)
        end = now - timedelta(days=30)
        start = end - timedelta(days=365*5)

        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sectorKey", info.get("quoteType", "uncategorized")).lower()
        ind = yf.Industry(info.get("industryKey")).name.lower() if info.get("industryKey") else str.lower(info.get("category")) if info.get("category") else "unknown"

        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
                cursor.execute("SELECT weight FROM ticker WHERE ticker = %s;", (ticker,))
                row = cursor.fetchone()
                
        if row is None:
            avgWeights = self._getIndustryAverageWeights(ind, sector)
            weight = [avgWeights, 0]
        else:
            try: weight = self.clean(row[0])
            except Exception: weight = [[0.2, 0.2, 0.2, 0.2, 0.2], 0]

        bestWeight = weight[0]

        history = stock.history(start=start-timedelta(days=730), end=end, interval="1d")
        if history.empty: return
        
        if history.index.tz is not None: history.index = history.index.tz_localize(None)
        window = history.loc[start.strftime('%Y-%m-%d'):end.strftime('%Y-%m-%d')]
        daily = window.resample("D").interpolate()
        if daily.index.tz is not None: daily.index = daily.index.tz_convert("America/New_York").tz_localize(None)
        origins = window["Close"].resample("1MS").last().dropna()

        bias = None
        weights = None
        errors = []
        def _getHistoricalData(origin):
            biasConfigs = {90:[bestWeight[0], "D"], 180:[bestWeight[1], "D"], 365:[bestWeight[2], "D"], 730:[bestWeight[3], "D"], 1825:[bestWeight[4], "D"]}
            res = self._forecast(stock, window, biasConfigs, origin, forward=90, parallel=True, uncertaintySamples=0)
            if res is None: return None
            rawCurves, rawSigmas, _ = res
            
            targetDates = [origin + timedelta(days=i) for i in range(90)]
            validIndices = []
            actuals = []
            for i, date in enumerate(targetDates):
                d = date.tz_convert("America/New_York").tz_localize(None) if date.tzinfo is not None else date
                if d in daily.index:
                    validIndices.append(i)
                    actuals.append(float(daily.loc[d, "Close"]))
            
            if not validIndices: return None
            return (rawCurves[:, validIndices], np.array(actuals))

        futures = {_THREAD_EXECUTOR.submit(_getHistoricalData, origin): origin for origin in origins.keys()}
        
        for future in as_completed(futures):
            origin = futures[future]
            resData = future.result()
            if resData is None: continue
            if userID: STATUS_REGISTRY[userID] = f"Backtesting for {str(origin.date().year)}..."
            
            matrix, targets = resData
            initGuess = np.array(bestWeight, dtype=float)
            initGuess = initGuess / np.sum(initGuess)
            
            res = minimize(self._smapeLoss, initGuess, args=(matrix, targets), method='SLSQP', bounds=self._BOUNDS, constraints=self._CONSTRAINTS)
            bestWeight = res.x.tolist()
            try: errors.append(float(res.fun))
            except Exception: pass
            
            prevWeights, trainingCount = weight
            adjustment = 0.05
            avgWeights = [prevWeights[j]*(1-adjustment) + bestWeight[j]*adjustment for j in range(len(prevWeights))]
            weights = [avgWeights, trainingCount+1]
        timestamp = str(math.floor(int(datetime.now().timestamp())))
        if weights is None: weights = weight

        def _serialize(o):
            if isinstance(o, np.generic): return o.item()
            if isinstance(o, (np.ndarray,)): return o.tolist()
            raise TypeError(f"Type {type(o)} not JSON serializable")

        serialized = json.dumps(weights, default=_serialize)
        avgError = float(sum(errors) / len(errors)) if len(errors) > 0 else 0.0

        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
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
                    (ticker, sector, ind, avgError, serialized, timestamp))
                DB_CONNECTION.commit()
        return weights

    def project(self, ticker, model, serverName, serverInvite, serverIcon, userID, lookbackStr="90d", overriddenWeights=None):
        recordRequest(userID, ticker)
        forward = 90
        
        # Parse lookback
        try:
            if lookbackStr == "ytd":
                lookback = 365
            else:
                lookback = int(lookbackStr.lower().replace("d",""))
        except:
            lookback = 90

        ticker = str(ticker).upper()
        stock = yf.Ticker(ticker)
        history = stock.history(period="5y", interval="1d") if model != 0 else stock.history(period="1wk")
        history = history.resample("D").interpolate(method="linear").ffill().bfill()
        if history.empty: return None
        
        curPrice = history["Close"].iloc[-1]
        lastDate = history.index[-1]
        plotHistory = history[history.index > lastDate - timedelta(days=lookback)]
        
        quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        futureDays = np.arange(0, forward + 1)

        bias = None
        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
                if userID: STATUS_REGISTRY[userID] = "Retrieving Latest Weights..."
                cursor.execute("select weight, updated from ticker where ticker = %s", (ticker,))
                rows = cursor.fetchone()

        points = []
        if model != 1:
            ivPoints = self._impliedVolatility(stock, lastDate, forward, curPrice, quantiles, futureDays)
            points = ivPoints if ivPoints is not None else []
        
        if model != 0:
            train = True if not bias else False
            if not bias and rows is not None:
                if len(json.dumps(rows)) > 1:
                    rows = self.clean(rows)
                    weight:list = rows[0]
                    updated = time.time()-int(rows[1])
                    if weight:
                        if updated < 432000:
                            bias = weight
                            train = False
            if train and not bias:
                isLeader = False
                with self._REGISTRY_LOCK:
                    if ticker not in self._TRAINING_REGISTRY:
                        event = threading.Event()
                        self._TRAINING_REGISTRY[ticker] = event
                        isLeader = True
                    else:
                        event = self._TRAINING_REGISTRY[ticker]
                
                if not isLeader:
                    if userID: STATUS_REGISTRY[userID] = f"Synchronizing {ticker} Prediction..."
                    event.wait()
                    with DB_LOCK:
                        with DB_CONNECTION.cursor() as curResult:
                            curResult.execute("select weight, updated from ticker where ticker = %s", (ticker,))
                            rows = curResult.fetchone()
                    if rows:
                        rows = self.clean(rows)
                        bias = rows[0]
                else:
                    try:
                        if userID: STATUS_REGISTRY[userID] = "Live Retraining Weights..."
                        bias = self._liveTrain(ticker=ticker, userID=userID)
                    finally:
                        with self._REGISTRY_LOCK:
                            event.set()
                            del self._TRAINING_REGISTRY[ticker]

            if overriddenWeights:
                bias = [overriddenWeights, 0]
            else:
                if bias is None or (not hasattr(bias, "__getitem__")) or len(bias) == 0:
                    info = stock.info
                    sector = info.get("sectorKey", info.get("quoteType", "uncategorized")).lower()
                    ind = yf.Industry(info.get("industryKey")).name.lower() if info.get("industryKey") else str.lower(info.get("category")) if info.get("category") else "unknown"
                    avgWeights = self._getIndustryAverageWeights(ind, sector)
                    bias = [avgWeights, 0]
            
            # Extract weights for ensemble component configuration
            currentWeights = bias[0]
            histories = {90: [currentWeights[0], "D"], 180: [currentWeights[1], "D"], 365: [currentWeights[2], "D"], 730: [currentWeights[3], "D"], 1825: [currentWeights[4], "D"]}
            
            startDate = lastDate - timedelta(days=lookback)
            window = history[history.index <= startDate]
            actuals = (history[(history.index > startDate) & (history.index <= lastDate)]["Close"].values)[:forward]
            
            if not overriddenWeights and len(actuals) > 20:
                raw, _, _ = self._forecast(stock, window, histories, startDate, forward=len(actuals), parallel=True)
                result = minimize(self._smapeLoss, currentWeights, args=(raw, actuals), method='SLSQP', bounds=self._BOUNDS, constraints=self._CONSTRAINTS)
                bestWeight = result.x
                # Re-apply weights to histories for the final forecast
                histories = {90: [bestWeight[0], "D"], 180: [bestWeight[1], "D"], 365: [bestWeight[2], "D"], 730: [bestWeight[3], "D"], 1825: [bestWeight[4], "D"]}
            else:
                bestWeight = np.array(currentWeights)

            if userID: STATUS_REGISTRY[userID] = "Applying Image Template..."

            future, futureSigma, insights = self._forecast(stock, history, histories, lastDate, forward=forward+1, parallel=True)
            if future is None: return None
            prophetTrend = np.dot(bestWeight, future)
            prophetSigma = np.dot(bestWeight, futureSigma)
            
            # Accountability: Calculate weighted global impacts from changepoints
            globalImpacts = {}
            for i, (deltas, cp_dates) in enumerate(insights):
                weight = bestWeight[i]
                for d, val in zip(cp_dates, deltas):
                    d_str = pd.Timestamp(d).strftime('%Y-%m-%d')
                    globalImpacts[d_str] = globalImpacts.get(d_str, 0) + (val * weight)
            # ------------------------------------------------
            
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
        
        idx50 = 5
        median = points[idx50]
        
        mid = len(quantiles) // 2
        for i in range(mid): ax.fill_between(futureDates, points[i], points[-(i+1)], color=themes.brand, alpha=0.15, lw=0)
        ax.plot(futureDates, median, color=themes.brand, linewidth=2, linestyle=("dashed" if model != 0 else "solid"))

        allDates = list(plotHistory.index) + futureDates
        self._formatAxes(ax, allDates, minY, maxY, median[-1], formatX=True)
        
        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax.annotate(f"${median[-1]:.2f}", xy=(1, median[-1]), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)
        ax.set_title(f"{str.upper(ticker)} Prediction (90d)", fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center")

        factors = []
        
        # Filter and rank Structural Shifts
        impactValues = np.array(list(globalImpacts.values()))
        if len(impactValues) > 0:
            threshold = np.std(impactValues) * 2.2 # 2.2 sigma for high significance
            sortedImpacts = sorted(globalImpacts.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for d_str, totalDelta in sortedImpacts:
                if abs(totalDelta) > threshold and len(factors) < 4:
                    symbol = themes.arrowUp if totalDelta > 0 else themes.arrowDown
                    color = themes.brand if totalDelta > 0 else themes.red
                    # Scale delta to a readable impact percentage (approximation)
                    factors.append({
                        "impact": {"symbol": symbol, "pct": f"{abs(totalDelta*10):.1f}%", "color": color, "val": totalDelta},
                        "label": f"Pattern shift on {pd.Timestamp(d_str).strftime('%x')}"
                    })
        factors = [f for f in factors if not (isinstance(f, dict) and f.get("impact", {}).get("pct") in ["0.0%", "0%"])]

        chartBuf = self._buffer(fig)
        if userID: STATUS_REGISTRY[userID] = "Finalizing Image..."
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon, styles="/predict", factors=factors).image(chartBuf), (median[-1], bestWeight.tolist())
    
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
                
                validIvs = pd.concat([calls["impliedVolatility"], puts["impliedVolatility"]])
                validIvs = validIvs[validIvs > 0.001]
                
                if validIvs.empty: continue
                meanIv = validIvs.mean()

                # even if daysDiff is 0 or 1, we force tYears to be at least 1/365; this prevents the square root of time from becoming 0 and collapsing the graph
                effectiveDays = max(daysDiff, 1.0)
                tYears = effectiveDays / 365.0
                
                expPrices = []
                for q in quantiles:
                    z = norm.ppf(q)
                    # Geometric Brownian Motion
                    projection = curPrice*np.exp(-1*meanIv**2*tYears+meanIv*np.sqrt(tYears)*z)+0.04
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
            
            if 2 <= span.days <= 150:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=24, min_n_ticks=16))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.tick_params(axis="x", rotation=90, colors=themes.grayDark, labelcolor=themes.grayDark, labelsize=10)
        
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
        ax.tick_params(axis="y", colors=themes.grayDark, labelcolor=themes.grayDark, labelsize=10)
        
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
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf

    def history(self, ticker, duration, interval, serverName, serverInvite, serverIcon, staticQuote, userID):
        recordRequest(userID, ticker)
        stock = yf.Ticker(ticker)
        periods = ["1d","5d","1mo","3mo","6mo","1y","ytd","2y","5y","10y","max"]
        intervals = ["2m","15m","30m","60m","1d","5d","1mo","3mo"]

        preview = duration
        if duration == "ytd" or duration == "1y":
            swaps = ["ytd","1y"]
            duration = swaps[not bool(swaps.index(duration))]

        if userID: STATUS_REGISTRY[userID] = "Retrieving Historical Data..."
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
                if periods.index(duration) > periods.index("1mo") and periods.index(duration) <= periods.index("2y"): string+="%b "
                if periods.index(duration) >= periods.index("5d") and periods.index(duration) <= periods.index("1mo"): string+="%a "
                if periods.index(duration) >= periods.index("1mo") and periods.index(duration) <= periods.index("ytd") : string+="%d "
                if periods.index(duration) > periods.index("1y") : string+="%Y "
                if intervals.index(interval) <= intervals.index("1d"): string+=(str(date.strftime("%I")).replace("0","")+":%M %p")
                return date.strftime(string)
            return ""

        history = stock.history(period=duration, interval=interval)
        if history.empty: return None

        if history.index.tz is None: history.index = history.index.tz_localize("UTC")
        history.index = history.index.tz_convert("America/New_York")

        fig = plt.figure(figsize=(20, 10), dpi=100)
        
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0) 
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
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
        
        volColors = [themes.brand if c >= o else themes.red for c, o in zip(history.Close, history.Open)]
        ax2.bar(history["x_index"], history.Volume, width=width, color=volColors, alpha=0.5)
        
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax1.spines["bottom"].set_color(themes.grayDark)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_color(themes.grayDark)
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
        ax2.set_xlim(-0.5, len(history)-0.5)

        ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        ax1.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)
        ax2.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        ax2.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)

        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax1.annotate(f"${lastPrice:.2f}", xy=(1, lastPrice), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)
        ax1.set_title(f"{str.upper(ticker)} History ({preview})", fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center", pad=20) 

        if userID: STATUS_REGISTRY[userID] = "Generating Chart..."
        chartBuf = self._buffer(fig)
        if userID: STATUS_REGISTRY[userID] = "Finalizing Chart..."
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon, styles="/chart", factors=staticQuote).image(chartBuf, displayLegend=False)

class User():
    def __init__(self, discordID):
        self._discordID = discordID

    def getAlerts(self):
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    account = self.accountFromDiscord(cursor=cursor)
                    if account:
                        cursor.execute("select symbol, price, updated from alert where account = %s", (account,))
                        return cursor.fetchall()
            return []
        except Exception:
            traceback.print_exc()
            return []

    def createAlert(self, symbol, price):
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    account = self.accountFromDiscord(cursor=cursor)
                    if account:
                        now = str(int(datetime.now().timestamp()))
                        cursor.execute(
                            "insert into alert (account, symbol, price, updated) values (%s, %s, %s, %s)",
                            (account, symbol.upper(), float(price), now)
                        )
                        DB_CONNECTION.commit()
                        return True
            return False
        except Exception:
            traceback.print_exc()
            return False

    def clearAlerts(self):
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    account = self.accountFromDiscord(cursor=cursor)
                    if account:
                        cursor.execute("delete from alert where account = %s", (account,))
                        DB_CONNECTION.commit()
                        return True
            return False
        except Exception:
            traceback.print_exc()
            return False

    def accountFromDiscord(self, cursor=None):
        if cursor is None:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cur:
                    cur.execute("select * from account where discord = %s", (str(self._discordID),))
                    row = cur.fetchone()
                    return row[0] if row is not None else None
        
        cursor.execute("select id from account where discord = %s", (str(self._discordID),))
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def getAnalytics(self):
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    accountID = self.accountFromDiscord(cursor=cursor)
                    if not accountID: return {"total": 0, "monthly": 0, "weekly": 0, "daily": 0}

                    now = int(time.time())
                    day = now - 86400
                    week = now - 604800
                    month = now - 2592000

                    # Total
                    cursor.execute("SELECT count(*) FROM request WHERE account = %s", (accountID,))
                    total = cursor.fetchone()[0]

                    # Daily
                    cursor.execute("SELECT count(*) FROM request WHERE account = %s AND (updated::bigint) > %s", (accountID, day))
                    daily = cursor.fetchone()[0]

                    # Weekly
                    cursor.execute("SELECT count(*) FROM request WHERE account = %s AND (updated::bigint) > %s", (accountID, week))
                    weekly = cursor.fetchone()[0]

                    # Monthly
                    cursor.execute("SELECT count(*) FROM request WHERE account = %s AND (updated::bigint) > %s", (accountID, month))
                    monthly = cursor.fetchone()[0]

                    return {"total": total, "monthly": monthly, "weekly": weekly, "daily": daily}
        except Exception:
            traceback.print_exc()
            return {"total": 0, "monthly": 0, "weekly": 0, "daily": 0}

    def createAccount(self, marketing:bool):
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    account = self.accountFromDiscord(cursor=cursor)
                    if account is None:
                        now = str(int(datetime.now().timestamp()))
                        cursor.execute(
                            "insert into account (discord, premium, preferences, credits, created, updated) values (%s, false, %s, 0, %s, %s) returning id",
                            (str(self._discordID), json.dumps({"marketing": marketing}), now, now)
                        )
                        returned = cursor.fetchone()
                        account = returned[0] if returned else None
                        DB_CONNECTION.commit()
                return account
        except Exception:
            traceback.print_exc()
            return None

def getAllAlerts():
    try:
        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
                cursor.execute('select a.id, ac.discord, a.symbol, a.price from alert a join account ac on a.account = ac.id')
                return cursor.fetchall()
    except Exception:
        traceback.print_exc()
        return []

def removeAlert(alertID):
    try:
        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
                cursor.execute('delete from alert where id = %s', (alertID,))
                DB_CONNECTION.commit()
                return True
    except Exception:
        traceback.print_exc()
        return False

def recordRequest(userID: int, tickerSymbol: str):
    tickerSymbol = tickerSymbol.upper()
    try:
        with DB_LOCK:
            with DB_CONNECTION.cursor() as cursor:
                # 1. Get/Create Ticker
                cursor.execute("SELECT id FROM ticker WHERE ticker = %s", (tickerSymbol,))
                row = cursor.fetchone()
                if row:
                    tickerID = row[0]
                else:
                    cursor.execute(
                        "INSERT INTO ticker (ticker, updated) VALUES (%s, '0') RETURNING id",
                        (tickerSymbol,)
                    )
                    tickerID = cursor.fetchone()[0]
                
                # 2. Get Account
                user = User(userID)
                accountID = user.accountFromDiscord(cursor=cursor)
                
                # 3. Log Request if account exists
                if accountID:
                    now = str(int(time.time()))
                    cursor.execute(
                        "INSERT INTO request (account, ticker, updated) VALUES (%s, %s, %s)",
                        (accountID, tickerID, now)
                    )
                    DB_CONNECTION.commit()
    except Exception:
        traceback.print_exc()

EPHEMERAL_FEEDBACK: dict[str, dict] = {}
EPHEMERAL_LOCK = threading.RLock()
DISLIKE_STRENGTH = 2

def getTickerFeedback(ticker: str):
    ticker = ticker.upper()
    likes, dislikes = 0, 0
    with EPHEMERAL_LOCK:
        if ticker in EPHEMERAL_FEEDBACK:
            data = EPHEMERAL_FEEDBACK[ticker]
            likes, dislikes = data.get("likes", 0), data.get("dislikes", 0)
    return likes, dislikes

def mutateWeights(ticker: str, currentWeights: list):
    likes, dislikes = getTickerFeedback(ticker)
    mutationStrength = 1.0 / (1.0 + (likes / (dislikes + 1.0)))
    
    # Stochastic Jump: Mix current weights with a random target
    randomTarget = np.random.rand(5)
    randomTarget /= randomTarget.sum()
    
    # Shift weights by the dynamic mutation strength
    mutated = [w * (1 - mutationStrength) + rt * mutationStrength for w, rt in zip(currentWeights, randomTarget)]
    total = sum(mutated)
    if total > 0: mutated = [w / total for w in mutated]
    return mutated

def recordPredictionFeedback(ticker: str, rating: str, currentWeights: list = None):
    ticker = ticker.upper()
    
    with EPHEMERAL_LOCK:
        if ticker not in EPHEMERAL_FEEDBACK:
            EPHEMERAL_FEEDBACK[ticker] = {"likes": 0, "dislikes": 0}
        
        if rating == "👍":
            EPHEMERAL_FEEDBACK[ticker]["likes"] += 1
        elif rating == "👎":
            EPHEMERAL_FEEDBACK[ticker]["dislikes"] += 1

    if rating == "👍" and currentWeights:
        try:
            with DB_LOCK:
                with DB_CONNECTION.cursor() as cursor:
                    cursor.execute("SELECT weight FROM ticker WHERE ticker = %s", (ticker,))
                    row = cursor.fetchone()
                    trainingCount = 0
                    if row and row[0]:
                        try:
                            decoded = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                            trainingCount = decoded[1] if len(decoded) > 1 else 0
                        except: pass

                    now = str(int(datetime.now().timestamp()))
                    serializedWeights = json.dumps([currentWeights, trainingCount])
                    
                    cursor.execute(
                        """
                        INSERT INTO ticker (ticker, weight, updated)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (ticker) DO UPDATE SET
                            weight = EXCLUDED.weight,
                            updated = EXCLUDED.updated;
                        """,
                        (ticker, serializedWeights, now)
                    )
                    DB_CONNECTION.commit()
                    with Charts._CACHE_LOCK:
                        Charts._CACHE.clear()
        except Exception:
            traceback.print_exc()

    return getTickerFeedback(ticker)
