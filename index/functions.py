import io
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from datetime import datetime, timedelta
from prophet import Prophet as ph
from pyfonts import set_default_font, load_google_font
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import themes
import requests
import ast

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
    def __init__(self):
        pass

    #Prediction methods
    def _impliedVolatility(self, stock, lastDate, forward, curPrice, quantiles, futureDays):
        anchorsY = [[curPrice] * len(quantiles)] 
        anchorsX = [0]
        
        options = stock.options
        if len(options) <= 1: return None

        #Ensure lastDate is treated as the 'start' of calc
        #If running at night, lastDate is likely the previous close
        start_date = lastDate.date()

        for exp in options:
            try:
                expDate = datetime.strptime(exp, "%Y-%m-%d").date()
                
                #Calculate raw days diff (avoids delta*sqrt(t) to be 0)
                daysDiff = (expDate - start_date).days
                
                #Skip past dates but allow 0 (today/tomorrow expirations) to be processed by clamping them to a minimum value below
                if daysDiff < 0: continue
                
                #Stop if too far into the future
                if daysDiff > forward + 15: break
                
                opt = stock.option_chain(exp)
                
                #Find ATM Strike
                centerStrike = curPrice
                calls = opt.calls.iloc[(opt.calls["strike"] - centerStrike).abs().argsort()[:2]]
                puts = opt.puts.iloc[(opt.puts["strike"] - centerStrike).abs().argsort()[:2]]
                
                #At night, IV can be 0.0 or NaN; if so, skip this expiration rather than breaking the whole model
                valid_ivs = pd.concat([calls["impliedVolatility"], puts["impliedVolatility"]])
                valid_ivs = valid_ivs[valid_ivs > 0.001] #Filter 0 or ~0
                
                if valid_ivs.empty: continue
                meanIV = valid_ivs.mean()

                #Even if daysDiff is 0 or 1, we force tYears to be at least 1/365; this prevents the square root of time from becoming 0 and collapsing the graph
                effective_days = max(daysDiff, 1.0)
                tYears = effective_days / 365.0
                
                expPrices = []
                for q in quantiles:
                    z = norm.ppf(q)
                    #Geometric Brownian Motion
                    projection = curPrice * np.exp(-0.5 * meanIV**2 * tYears + meanIV * np.sqrt(tYears) * z)
                    expPrices.append(projection)
                
                #Use the actual daysDiff for plotting the X-axis anchor even if we used a floor for the math
                anchorsX.append(max(daysDiff, 1)) #visual anchor minimum 1 day out
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
    
    def _prophetInit(self, model, history, lastDate, curPrice):
        forward = 90
        prophetTrend = None
        prophetSigma = 0
        if model != 0:
            prophetSum = []
            histories = {90: [0.005, "ME"], 180: [0.01, "ME"], 365: [0.485, "D"], 730: [0.49, "W"], 1825: [0.01, "YS"]} # days: [weight, freq] #weights must be = 1
            
            for h, nested in histories.items():
                start_date = lastDate - timedelta(days=h)
                data = history[history.index > start_date].reset_index()[["Date", "Close"]]
                if len(data) < 20: continue

                data.columns = ["ds", "y"]
                data["ds"] = data["ds"].dt.tz_localize(None)

                m = ph(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
                m.fit(data)
                
                future = m.make_future_dataframe(periods=forward, freq=nested[1]) # dynamic intraday
                fcst = m.predict(future)
                
                trend = fcst.tail(forward + 1)["yhat"].values
                #Offset to align with current price
                prophetSum.append((trend + curPrice - trend[0]) * nested[0])
            
            if prophetSum:
                prophetTrend = np.sum(prophetSum, axis=0)
                
        return prophetTrend, prophetSigma
    
    def _prophetTest(self, history, lastDate, curPrice, histories):
        # {30: [0.25, "D"], 365: [0.25, "W"], 730: [0.2, "ME"], 1095: [0.2, "ME"], 1825: [0.05, "YE"]} # days: [weight, freq] #weights must be = 1
        forward = 90
        prophetTrend = None
        prophetSigma = 0

        def parse_weight_freq_map(s: str) -> dict[int, list]:
            # Convert "D" → 'D' so literal_eval can parse it
            fixed = s.replace('"', "'")
            
            # Safely evaluate into Python objects
            return ast.literal_eval(fixed)

        histories = parse_weight_freq_map(histories)
        prophetSum = []
            
        for h, nested in histories.items():
            start_date = lastDate - timedelta(days=h)
            data = history[history.index > start_date].reset_index()[["Date", "Close"]]
            if len(data) < 20: continue

            data.columns = ["ds", "y"]
            data["ds"] = data["ds"].dt.tz_localize(None)

            m = ph(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
            m.fit(data)
            
            future = m.make_future_dataframe(periods=forward, freq=nested[1]) # dynamic intraday
            fcst = m.predict(future)
            
            trend = fcst.tail(forward + 1)["yhat"].values
            #Offset to align with current price
            prophetSum.append((trend + curPrice - trend[0]) * nested[0])
        
        if prophetSum:
            prophetTrend = np.sum(prophetSum, axis=0)
                
        return prophetTrend, prophetSigma

    #Plotting encapsulation
    #Reverted to single axis return, as we are now doing an overlay
    def _setup_figure(self):
        plt.rc("font", size=10)
        fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
        fig.patch.set_facecolor(color=themes.bgDark)
        ax.set_facecolor(themes.bgDark)
        return fig, ax
    
    def _format_axes(self, ax, dates, minY, maxY, lastPrice=None, formatX=True):
        #X Axis: dates
        if formatX:
            #Dynamic Date Formatting based on span
            span = dates[-1] - dates[0]
            if span.days > 730: # > 2 Years
                fmt = "%Y"
            elif dates[-1].year != dates[0].year: #Spans across a new year
                fmt = "%b %Y"
            else:
                fmt = "%b %d"
            
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.tick_params(axis="x", rotation=45, colors=themes.grayDark, labelcolor=themes.grayDark)
        
        #Price Auto-Scale
        yRange = maxY - minY
        if yRange == 0: yRange = 1
        rawStep = yRange / 20
        allowedSteps = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        step = min(allowedSteps, key=lambda x: abs(x - rawStep))
        
        if lastPrice:
            ticksUp = np.arange(lastPrice, maxY * 1.05, step)
            ticksDown = np.arange(lastPrice - step, minY * 0.95, -step)
            customTicks = np.sort(np.concatenate((ticksDown, ticksUp)))
        else:
            customTicks = np.arange(minY, maxY, step)

        ax.set_yticks(customTicks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
        
        #FORCE PRICE TO RIGHT
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", colors=themes.grayDark, labelcolor=themes.grayDark)
        
        #Styling
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_color(themes.grayDark)
        ax.spines["bottom"].set_color(themes.grayDark)
        
        #Grid
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        if formatX:
            ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)
        
        ax.set_ylim(minY * 0.98, maxY * 1.02)
        if formatX:
            ax.set_xlim(dates[0], dates[-1])

    def _draw_gradient(self, ax, xNums, yVals, minY, color):
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

    def _save_buffer(self, fig):
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf

    #Chart history
    def history(self, ticker, duration, serverName, serverInvite, serverIcon):
        stock = yf.Ticker(ticker)
        
        interval = "1d"
        if duration in ["1d"]: interval = "2m"
        elif duration in ["5d"]: interval = "15m"
        elif duration in ["1mo"]: interval = "1d"
        elif duration in ["3mo", "6mo"]: interval = "1wk"
        elif duration in ["1y", "2y"]: interval = "1mo"
        else: interval = "3mo"

        history = stock.history(period=duration, interval=interval)
        if history.empty: return None

        # Timezone Fix
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC")
        history.index = history.index.tz_convert("America/New_York")

        # Setup Main Axis (Price)
        fig, ax1 = self._setup_figure()
        
        # Create Twin Axis (Volume) sharing X
        ax2 = ax1.twinx()

        # Integer Indexing
        history = history.copy()
        history["x_index"] = np.arange(len(history))
        
        up = history[history.Close >= history.Open]
        down = history[history.Close < history.Open]
        width = 0.6
        width2 = 0.08 

        # --- Plot Volume (ax2) - Bottom Layer ---
        maxVol = history.Volume.max()
        ax2.set_ylim(0, maxVol * 4) 
        
        vol_colors = [themes.brand if c >= o else themes.brandInvert for c, o in zip(history.Close, history.Open)]
        ax2.bar(history["x_index"], history.Volume, width=width, color=vol_colors, alpha=0.5)
        
        # Configure Volume Axis (Left)
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_color(themes.grayDark)
        ax2.tick_params(axis="y", colors=themes.grayDark, labelcolor=themes.grayDark, labelsize=8)
        
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        def vol_format(x, pos): return Humanizer.suffix(x)
        ax2.yaxis.set_major_formatter(FuncFormatter(vol_format))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=50))

        # --- Plot Price (ax1) - Top Layer ---
        ax1.set_zorder(10)
        ax1.patch.set_visible(False)

        ax1.bar(up["x_index"], up.Close - up.Open, bottom=up.Open, width=width, color=themes.brand)
        ax1.bar(up["x_index"], up.High - up.Close, bottom=up.Close, width=width2, color=themes.brand)
        ax1.bar(up["x_index"], up.Low - up.Open, bottom=up.Open, width=width2, color=themes.brand)
        
        downColor = themes.brandInvert
        ax1.bar(down["x_index"], down.Close - down.Open, bottom=down.Open, width=width, color=downColor)
        ax1.bar(down["x_index"], down.High - down.Open, bottom=down.Open, width=width2, color=downColor)
        ax1.bar(down["x_index"], down.Low - down.Close, bottom=down.Close, width=width2, color=downColor)

        #Limits
        minY = history["Low"].min()
        maxY = history["High"].max()
        lastPrice = history["Close"].iloc[-1]

        #X-Axis Formatter (Integer -> Date)
        def format_date(x, pos=None):
            idx = int(x)
            if 0 <= idx < len(history):
                date_val = history.index[idx]
                if duration == "1d": 
                    return date_val.strftime("%H:%M")
                elif duration == "5d": 
                    return date_val.strftime("%b %d\n%H:%M")
                elif duration in ["1mo", "3mo", "6mo", "ytd"]:
                    return date_val.strftime("%b %d")
                elif duration in ["1y", "2y"]:
                    return date_val.strftime("%b %Y") #Month Year
                else: 
                    return date_val.strftime("%Y") #Just Year
            return ""

        ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax1.tick_params(axis="x", colors=themes.grayDark, labelcolor=themes.grayDark)
        
        self._format_axes(ax1, history["x_index"].values, minY, maxY, lastPrice, formatX=False)
        ax1.set_xlim(-0.5, len(history) - 0.5)

        #ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        ax1.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark)
        #ax1.set_axisbelow(True) 

        #Annotation
        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax1.annotate(f"${lastPrice:.2f}", xy=(1, lastPrice), xycoords=("axes fraction", "data"), 
                    xytext=(5, 0), textcoords="offset points", va="center", ha="left", 
                    color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)

        plt.title(f"{str.upper(ticker)} History ({duration})", 
                  fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center")

        chartBuf = self._save_buffer(fig)
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon).image(chartBuf, displayLegend=False)
    
    def project(self, ticker, model, serverName, serverInvite, serverIcon):
        forward = 90
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo") if model == 0 else stock.history(period="5y", interval="1d")
        if history.empty: return None
        
        curPrice = history["Close"].iloc[-1]
        lastDate = history.index[-1]
        
        #Data Slicing for plotting context
        plotHistory = history[history.index > lastDate - timedelta(days=14)] if model != 0 else history
        quantiles = np.linspace(0.05, 0.95, 11)
        futureDays = np.arange(0, forward + 1)
        
        points = []
        prophetTrend, prophetSigma = self._prophetInit(model, history, lastDate, curPrice)

        if model != 1: #Calculate IV if not purely Prophet
            ivPoints = self._impliedVolatility(stock, lastDate, forward, curPrice, quantiles, futureDays)
            points = ivPoints if ivPoints is not None else []
            
        if model == 1: #Prophet Only
            if prophetTrend is None: raise ValueError("Prophet generation failed")
            points = np.array([prophetTrend + (norm.ppf(q) * prophetSigma) for q in quantiles])
            
        elif model == 2 and len(points) > 0 and prophetTrend is not None: #Aggregate
            spread = points - curPrice
            points = np.array([prophetTrend + spread[i] for i in range(len(quantiles))])

        if len(points) == 0: return None #Return None to redo model using Prophet only

        #Plotting
        points = np.maximum(points, 0.01)
        futureDates = [lastDate + timedelta(days=int(d)) for d in futureDays]
        
        fig, ax = self._setup_figure()
        
        #Draw Line + Gradient
        ax.plot(plotHistory.index, plotHistory["Close"], color=themes.brand, linewidth=2, zorder=10)
        
        minY = min(plotHistory["Close"].min(), np.min(points))
        maxY = max(plotHistory["Close"].max(), np.max(points))
        
        self._draw_gradient(ax, mdates.date2num(plotHistory.index), plotHistory["Close"].values, minY, themes.brand)
        
        #Draw Fan
        mid = len(quantiles) // 2
        for i in range(mid):
            ax.fill_between(futureDates, points[i], points[-(i+1)], color=themes.brand, alpha=0.15, lw=0)

        #Draw Median Line
        median = points[mid]
        ax.plot(futureDates, median, color=themes.brand, linewidth=2, linestyle=("dashed" if model != 0 else "solid"))

        #Format & Save
        allDates = list(plotHistory.index) + futureDates
        self._format_axes(ax, allDates, minY, maxY, median[-1])
        
        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax.annotate(f"${median[-1]:.2f}", xy=(1, median[-1]), xycoords=("axes fraction", "data"), 
                    xytext=(5, 0), textcoords="offset points", va="center", ha="left", 
                    color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)
        
        plt.title(f"{str.upper(ticker)} Prediction (90d)", 
                  fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center")

        chartBuf = self._save_buffer(fig)
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon).image(chartBuf)
    
    def projectTest(self, ticker, weights, today): #period given in days
        today = datetime.strptime(today, "%Y-%m-%d")
        forward = 90
        stock = yf.Ticker(ticker)
        history = stock.history(start=today-timedelta(days=365), end=today, interval="1d")
        if history.empty: return None
        
        curPrice = history["Close"].iloc[-1]
        lastDate = history.index[-1]
        
        #Data Slicing for plotting context
        plotHistory = history[history.index > lastDate-timedelta(days=14)]
        quantiles = np.linspace(0.05, 0.95, 11)
        futureDays = np.arange(0, forward + 1)
        
        points = []
        prophetTrend, prophetSigma = self._prophetTest(history=history, lastDate=lastDate, curPrice=curPrice, histories=weights)
        if prophetTrend is None: raise ValueError("Prophet generation failed")
        points = np.array([prophetTrend + (norm.ppf(q) * prophetSigma) for q in quantiles])

        #Plotting
        points = np.maximum(points, 0.01)
        futureDates = [lastDate + timedelta(days=int(d)) for d in futureDays]
        
        fig, ax = self._setup_figure()
        
        #Draw Line + Gradient
        ax.plot(plotHistory.index, plotHistory["Close"], color=themes.brand, linewidth=2, zorder=10)
        
        minY = min(plotHistory["Close"].min(), np.min(points))
        maxY = max(plotHistory["Close"].max(), np.max(points))
        
        self._draw_gradient(ax, mdates.date2num(plotHistory.index), plotHistory["Close"].values, minY, themes.brand)
        
        #Draw Fan
        mid = len(quantiles) // 2
        for i in range(mid):
            ax.fill_between(futureDates, points[i], points[-(i+1)], color=themes.brand, alpha=0.15, lw=0)

        #Draw Median Line
        median = points[mid]
        ax.plot(futureDates, median, color=themes.brand, linewidth=2, linestyle=("dashed"))

        #Format & Save
        allDates = list(plotHistory.index) + futureDates
        self._format_axes(ax, allDates, minY, maxY, median[-1])
        
        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax.annotate(f"${median[-1]:.2f}", xy=(1, median[-1]), xycoords=("axes fraction", "data"), 
                    xytext=(5, 0), textcoords="offset points", va="center", ha="left", 
                    color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox)
        
        plt.title(f"{str.upper(ticker)} Prediction (90d)", 
                  fontdict={"weight": "black", "size": 40, "color": themes.brand}, loc="center")

        chartBuf = self._save_buffer(fig)
        return chartBuf