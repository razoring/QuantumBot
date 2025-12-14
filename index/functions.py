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

from themes import brand, bgDark
# end of imports

matplotlib.use("Agg") # set backend / disables ui opening
logging.getLogger("prophet").setLevel(logging.WARNING) # pre setup / disable logging
logging.getLogger("cmdstanpy").disabled = True

class yFinanceWrapper:
    pass

class Projection:
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
            histories = {365:0.5, 730:0.15, 1095:0.15, 1825:0.2} #years in days: weight
            for h in histories:
                data = history[history.index > lastDate - timedelta(days=h)].reset_index()[["Date", "Close"]]
                data.columns = ["ds", "y"]
                data["ds"] = data["ds"].dt.tz_localize(None)

                prophet = ph(daily_seasonality=True, yearly_seasonality=True)
                prophet.fit(data)

                future = prophet.make_future_dataframe(periods=forward, freq="D") # match freq
                fcst = prophet.predict(future)

                trend = fcst.tail(forward + 1)["yhat"].values
                prophetSum.append((trend+curPrice - trend[0]) * histories.get(h))
            prophetTrend = np.sum(prophetSum, axis=0)
        return prophetTrend, prophetSigma

    def _createPlot(self, model, history, lastDate, futureDays, quantiles, points):
        plotHistory = history[history.index > lastDate - timedelta(days=7)]
        futureDates = [lastDate + timedelta(days=int(d)) for d in futureDays]

        # floor points
        points = np.maximum(points, 0.01)
        # plot the graph
        plt.rc("font", weight="bold", size=10)
        fig, ax = plt.subplots(figsize=(20, 10), dpi=120)
        fig.patch.set_facecolor(color=bgDark)
        ax.plot(plotHistory.index, plotHistory["Close"], color=brand, linewidth=2, zorder=10)
        minY = min(plotHistory["Close"].min(), np.min(points))
        maxY = max(plotHistory["Close"].max(), np.max(points))
        xNums = mdates.date2num(plotHistory.index)
        yVals = plotHistory["Close"].values
        yFloor = minY * 0.90
        
        #gradient logic
        verts = [(xNums[0], yFloor)] + list(zip(xNums, yVals)) + [(xNums[-1], yFloor)]
        poly = Polygon(verts, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(poly)
        cTop = to_rgba(brand, alpha=0.3)
        cBot = to_rgba(brand, alpha=0.0)
        gradientCmap = LinearSegmentedColormap.from_list("history_gradient", [cBot, cTop])
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        im = ax.imshow(gradient, aspect="auto", cmap=gradientCmap, origin="lower", extent=[xNums[0], xNums[-1], yFloor, yVals.max()], zorder=1)
        im.set_clip_path(poly)
        
        # fan graph
        mid = len(quantiles) // 2
        for i in range(mid):
            lower_curve = points[i]
            upper_curve = points[-(i+1)]
            ax.fill_between(futureDates, lower_curve, upper_curve, color=brand, alpha=0.15, lw=0)

        # legend
        elements = []
        if model != 1:
            for i in range(0, mid, 1): 
                q = quantiles[i]
                ci = int(round(50-((1 - 2*q) * 50)))
                simulated_alpha = 1 - (1 - 0.15) ** (i + 1)
                elements.append(Patch(facecolor=brand, edgecolor=None, alpha=simulated_alpha, label=f"{ci}% Probability"))
        elements.append(Line2D([0], [0], color=brand, lw=2, label=("50% Probability" if model != 1 else "Prediction"), linestyle= ("dashed" if model == 1 else "solid")))
        leg = ax.legend( handles=elements, loc="lower left", facecolor=bgDark, edgecolor="gray", framealpha=1.0, fancybox=True, labelcolor="white", fontsize=8, borderpad=0.8)
        leg.get_frame().set_linewidth(1)

        # 50% line
        median = points[mid] # make them start at the same spot
        ax.plot(futureDates, median, color=brand, linewidth=2, linestyle= ("dashed" if model == 1 else "solid"))

        # labels
        ax = plt.gca()
        ax.set_facecolor(bgDark)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=90, colors="gray")
        #plt.setp(ax.get_xticklabels(), weight="bold")
        leg.get_frame().set_linewidth(1)
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
        ax.tick_params(axis="y", colors="gray")
        bbox = dict(boxstyle="square,pad=0.3", fc=bgDark, ec="none", alpha=1.0)
        ax.annotate(f"${median[-1]:.2f}", xy=(1, median[-1]), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=brand, fontweight="bold", fontsize=11, bbox=bbox,)
        #ax.text(futureDates[-1], median[-1], f" ${median[-1]:.2f}", color=colour, fontweight="bold", fontsize=11, va="center", ha="left")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_color("gray")
        ax.spines["bottom"].set_color("gray")
        
        # grid
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
        ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.3) # added x-grid to see days better
        plt.ylim(minY * 0.98, maxY * 1.02)
        
        # combine both line and fan graphs
        dates = list(plotHistory.index) + futureDates
        plt.xlim(dates[0], dates[-1])

        plt.tight_layout()
        # save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0) # rewind buffer

        return buf

    def create(self, ticker, model):
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
        
        return self._createPlot(model=model, history=history, lastDate=lastDate, futureDays=futureDays, quantiles=quantiles, points=points)