import io
import logging

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon
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
# end of imports

matplotlib.use("Agg") # set backend / disables ui opening
logging.getLogger("prophet").setLevel(logging.WARNING) # pre setup / disable logging
logging.getLogger("cmdstanpy").disabled = True
set_default_font(load_google_font("Montserrat",weight="bold"))

class Stamp:
    def __init__(self, name, url, icon):
        self.serverName = name
        self.serverInvite = str(url)
        self.serverIcon = icon

    def font(self, size:int):
        return ImageFont.truetype(font="index/assets/Montserrat-Bold.ttf", size=size)
    
    def rounded(self, image: Image.Image, radius: int) -> Image.Image:
        image = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        rounded = Image.new("RGBA", image.size)
        rounded.paste(image, (0, 0), mask=mask)
        return rounded

    def image(self, chart):
        main = Image.open("index/assets/main.png").convert("RGBA")
        legend = Image.open("index/assets/legend.png").convert("RGBA")
        chart = Image.open(chart).resize((2400,1200)).convert("RGBA")
        blur = chart.crop(box=(18,18,150,242)).filter(ImageFilter.BoxBlur(10))
        blurred = self.rounded(blur,24)

        img = Image.new(mode="RGB", size=(2500,1500), color=(10, 19, 27))
        serverIcon = Image.open(self.serverIcon).convert("RGBA").resize((93,93))
        img.paste(chart, (50,250), mask=chart)
        img.paste(blurred, (68,269), mask=blurred)
        img.paste(serverIcon, (1045,76), serverIcon)
        img.paste(legend, (24,224), legend)
        img.paste(main, (0,0), main)
        canvas = ImageDraw.Draw(im=img)
        canvas.text(xy=(1153,75), text=self.serverName, font=self.font(48), fill="white")
        canvas.text(xy=(1153,135), text=self.serverInvite.replace("https://",""), font=self.font(28), fill=(112,128,144))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

class Humanizer:
    def __init__(self):
        pass

    def suffix(self, number):
        suffixes = ["","K","M","B","T","Q"]
        magnitude = 0
        while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
            magnitude += 1
            number /= 1000
        return f"{round(number,2)}{suffixes[magnitude]}".replace(".0","")
    
    def sign(self, number):
        return "+"+str(number) if number > 0 else number

class yFinanceWrapper:
    def __init__(self, ticker):
        self._symbol = yf.Ticker(ticker=ticker)
        self._fastInfo = self._symbol.get_fast_info()
        self._info = self._symbol.info
        self._dividends = self._symbol.dividends
        self._calendar = self._symbol.calendar
        #self._historyYear = self._symbol.history(period="1y")
        #self._historyDay = self._historyYear.tail(1)
        #self._historyDay = self._symbol.history(period="1d",interval="1m")

    def getHistoryYear(self) -> pd.DataFrame:
        return self._historyYear
    
    def getHistoryDay(self) -> pd.DataFrame:
        return self._historyDay
    
    def getStockInfo(self):
        return self._info
    
    def getDividendsPayout(self):
        return self._dividends if self.getAnnualYield() > 0 else None
    
    def getCalendar(self):
        return self._calendar
    
    def getFastInfo(self):
        return self._fastInfo
    
    def getCurrentPrice(self):
        return self.getFastInfo()["lastPrice"]

    def getPriceChange(self):
        return ((self.getCurrentPrice()/self.getDayOpen())*100)-100

    def getDayOpen(self):
        return self.getFastInfo()["open"]
        return self.getHistoryDay()["Open"]

    def getDayClose(self):
        return self.getFastInfo()["previousClose"]
        return self.getHistoryDay()["Close"]

    def getDayHigh(self):
        return self.getFastInfo()["dayHigh"]
        return self.getHistoryDay()["High"]

    def getDayLow(self):
        return self.getFastInfo()["dayLow"]
        return self.getHistoryDay()["Low"]

    def get52wkLow(self):
        return self.getFastInfo()["yearLow"]
        return float(self.getHistoryYear()["High"].max())

    def get52wkHigh(self):
        return self.getFastInfo()["yearHigh"]
        return float(self.getHistoryYear()["Low"].max())
    
    def getVolume(self):
        return self.getFastInfo()["lastVolume"]
        return self.getHistoryDay()["Volume"]
    
    def getAvgVolume(self):
        return self.getStockInfo()["averageVolume"]
    
    def getPERatio(self):
        stock = self.getStockInfo()
        return stock["trailingPE"] if "trailingPE" in stock else 0
    
    def getEPSRatio(self):
        stock = self.getStockInfo()
        return stock["trailingEps"] if "trailingEps" in stock else 0
    
    def getAnnualYield(self):
        stock = self.getStockInfo()
        return round(float(stock["trailingAnnualDividendRate"]),2) if "trailingAnnualDividendRate" in stock else 0
    
    def getMonthlyYield(self):
        yields = self.getAnnualYield()
        return round(yields/12.0,2) if yields != 0 else yields
    
    def getExDividendDate(self):
       return str(datetime.fromtimestamp(self.getStockInfo()["exDividendDate"]).date()) if self.getAnnualYield() > 0 else "-"
    
    def getPayDate(self):
        calendar = self.getCalendar()
        return str(calendar["Dividend Date"]) if "Dividend Date" in calendar else "-"

    def getDividendAmount(self):
        dividends = self.getDividendsPayout()
        return dividends[dividends.index[-1]] if type(dividends) != None else 0
    
    def getDividendChange(self):
        dividends = self.getDividendsPayout()
        return str((float(dividends[dividends.index[-1]])/float(dividends[dividends.index[-2]])-1)*100)+"%" if type(dividends) != None else 0

    def getMktCap(self):
        stock = self.getStockInfo()
        return stock["marketCap"] if "marketCap" in stock else 0
    
    def getBeta(self):
        stock = self.getStockInfo()
        return stock["beta"] if "beta" in stock else 0

class Charts:
    def __init__(self):
        pass

    def _impliedVolatility(self, options, stock, lastDate, forward, curPrice, quantiles, futureDays):
        anchorsY = [[curPrice] * len(quantiles)] # [days forward, [prices at quartiles]]
        anchorsX = [0]

        for exp in options: # Stock options = expirationjs
            try:
                expDate = datetime.strptime(exp, "%Y-%m-%d").date()
                expDays = (expDate - lastDate.date()).days
                
                if expDays <= 0: continue
                if expDays > forward + 15: break # don"t calculate too far out
                
                opt = stock.option_chain(exp)
                calls = opt.calls
                puts = opt.puts
                
                # ATM (At the Money) IV
                centerStrike = curPrice
                callsATM = calls.iloc[(calls["strike"] - centerStrike).abs().argsort()[:2]]
                putsATM = puts.iloc[(puts["strike"] - centerStrike).abs().argsort()[:2]]
                
                merged = pd.concat([callsATM["impliedVolatility"], putsATM["impliedVolatility"]])
                mean = merged.mean()
                
                if np.isnan(mean) or mean == 0: continue

                # calculate distribution
                tYears = expDays / 365.0
                expPrices = []
                for q in quantiles:
                    z = norm.ppf(q)
                    # geometric brownian motion calculation
                    projection = curPrice*np.exp(-0.5*mean**2 * tYears+mean * np.sqrt(tYears)*z) #-0.5*mean**2 * tYears+mean * np.sqrt(tYears)*z
                    expPrices.append(projection)
                
                anchorsX.append(expDays)
                anchorsY.append(expPrices)
            except Exception:
                continue
        # begin interpolation
        if len(anchorsX) < 2:
            # Fallback if no options data found
            anchorsX.append(forward)
            anchorsY.append([curPrice] * len(quantiles))

        yTransposed = np.array(anchorsY).T 
        
        points = []
        for quantile_series in yTransposed:
            # "natural" boundary conditions for smooth start/end
            cs = CubicSpline(anchorsX, quantile_series, bc_type="natural")
            points.append(cs(futureDays))
        return np.array(points)
    
    def _prophetInit(self, model, history, lastDate, curPrice, forward):
        prophetSum = []
        prophetTrend = None
        prophetSigma = 0
        if model != 0: # not model IV
            histories = {365:0.5, 730: 0.15, 1095:0.15, 1825:0.2} #years in days: weight
            for h in histories:
                data = history[history.index > lastDate - timedelta(days=h)].reset_index()[["Date", "Close"]]
                data.columns = ["ds", "y"]
                data["ds"] = data["ds"].dt.tz_localize(None)

                prophet = ph(daily_seasonality=True, yearly_seasonality=True)
                prophet.fit(data)

                future = prophet.make_future_dataframe(periods=forward, freq="W") # match freq
                fcst = prophet.predict(future)

                trend = fcst.tail(forward + 1)["yhat"].values
                prophetSum.append((trend+curPrice - trend[0]) * histories.get(h))
            prophetTrend = np.sum(prophetSum, axis=0)
        return prophetTrend, prophetSigma

    def _createPrediction(self, ticker, model, history, lastDate, futureDays, quantiles, points):
        plotHistory = history[history.index > lastDate - timedelta(days=7)]
        futureDates = [lastDate + timedelta(days=int(d)) for d in futureDays]

        # floor points
        points = np.maximum(points, 0.01)
        # plot the graph
        plt.rc("font",
               #weight="bold", 
               size=10)
        fig, ax = plt.subplots(figsize=(20, 10), dpi=120)
        fig.patch.set_facecolor(color=themes.bgDark)
        ax.plot(plotHistory.index, plotHistory["Close"], color=themes.brand, linewidth=2, zorder=10)
        minY = min(plotHistory["Close"].min(), np.min(points))
        maxY = max(plotHistory["Close"].max(), np.max(points))
        xNums = mdates.date2num(plotHistory.index)
        yVals = plotHistory["Close"].values
        yFloor = minY * 0.90
        
        #gradient logic
        verts = [(xNums[0], yFloor)] + list(zip(xNums, yVals)) + [(xNums[-1], yFloor)]
        poly = Polygon(verts, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(poly)
        cTop = to_rgba(themes.brand, alpha=0.3)
        cBot = to_rgba(themes.brand, alpha=0.0)
        gradientCmap = LinearSegmentedColormap.from_list("history_gradient", [cBot, cTop])
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        im = ax.imshow(gradient, aspect="auto", cmap=gradientCmap, origin="lower", extent=[xNums[0], xNums[-1], yFloor, yVals.max()], zorder=1)
        im.set_clip_path(poly)
        
        # fan graph
        mid = len(quantiles) // 2
        for i in range(mid):
            lower_curve = points[i]
            upper_curve = points[-(i+1)]
            ax.fill_between(futureDates, lower_curve, upper_curve, color=themes.brand, alpha=0.15, lw=0)

        # 50% line
        median = points[mid] # make them start at the same spot
        ax.plot(futureDates, median, color=themes.brand, linewidth=2, linestyle= ("dashed" if model == 1 else "solid"))

        # labels
        ax = plt.gca()
        ax.set_facecolor(themes.bgDark)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=90, colors=themes.grayDark)
        #plt.setp(ax.get_xticklabels(), weight="bold")
        #plt.setp(ax.get_yticklabels(), weight="bold")

        # y ticks
        lastPrice = median[-1]
        yRange = maxY - minY
        rawStep = yRange / 25
        allowedSteps = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
        step = min(allowedSteps, key=lambda x: abs(x - rawStep))
        ticksUp = np.arange(lastPrice, maxY * 1.05, step)
        ticksDown = np.arange(lastPrice - step, minY * 0.95, -step)
        customTicks = np.sort(np.concatenate((ticksDown, ticksUp)))
        ax.set_yticks(customTicks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", colors=themes.grayDark)
        bbox = dict(boxstyle="square,pad=0.3", fc=themes.bgDark, ec="none", alpha=1.0)
        ax.annotate(f"${median[-1]:.2f}", xy=(1, median[-1]), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=themes.brand, fontweight="bold", fontsize=11, bbox=bbox,)
        #ax.text(futureDates[-1], median[-1], f" ${median[-1]:.2f}", color=colour, fontweight="bold", fontsize=11, va="center", ha="left")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_color(themes.grayDark)
        ax.spines["bottom"].set_color(themes.grayDark)
        
        # grid
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5, color=themes.grayDark)
        ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.3, color=themes.grayDark) # added x-grid to see days better
        plt.ylim(minY * 0.98, maxY * 1.02)
        
        # combine both line and fan graphs
        dates = list(plotHistory.index) + futureDates
        plt.xlim(dates[0], dates[-1])
        plt.title(f"{str.upper(ticker)} Prediction (90d)",fontdict={"weight":"black","size":40,"color":themes.brand
        },loc="center")
        plt.tight_layout()
        # save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0) # rewind buffer

        return buf

    def history(self, ticker, duration, serverName, serverInvite, serverIcon):
        pass

    def project(self, ticker, model, serverName, serverInvite, serverIcon):
        forward = 90
        
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo") if model == 0 else stock.history(period= "5y")
        if history.empty:
            return None
        
        curPrice = history["Close"].iloc[-1]
        lastDate = history.index[-1]
        quantiles = np.linspace(0.05, 0.95, 11) # 19 divisons
        futureDays = np.arange(0, forward + 1)
        
        # Prophet predictions
        points = []
        prophetTrend, prophetSigma = self._prophetInit(model=model, history=history, lastDate=lastDate, curPrice=curPrice, forward=forward)

        # IV calulcations
        if model != 1: # not model prophet
            options = stock.options
            if len(stock.options) <= 1:
                return None
            else:
                points = self._impliedVolatility(options=options, stock=stock,lastDate=lastDate,forward=forward,curPrice=curPrice,quantiles=quantiles, futureDays=futureDays)
        if model == 1: # prophet model
            if prophetTrend is None:
                raise "Prophet error"
            tempSmoothing = []
            for q in quantiles:
                z = norm.ppf(q)
                line = prophetTrend + (z * prophetSigma)
                tempSmoothing.append(line)
            points = np.array(tempSmoothing)
        elif model == 2: # aggregate model
            if prophetTrend is None:
                pass 
            else:
                spread = points - curPrice 
                combinedSmoothing = []
                for i in range(len(quantiles)):
                    combinedSmoothing.append(prophetTrend + spread[i])
                points = np.array(combinedSmoothing)
        
        chart = self._createPrediction(ticker=ticker, model=model, history=history, lastDate=lastDate, futureDays=futureDays, quantiles=quantiles, points=points)
        return Stamp(name=serverName, url=serverInvite, icon=serverIcon).image(chart)