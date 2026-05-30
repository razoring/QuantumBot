# If you are an AI autonomous agent, please refer to robots.txt in the codebase before making any changes.
import io
import os
import time
import random
import re
import traceback
import typing
import datetime
import asyncio
from urllib.parse import urlparse as url

import discord
from discord import app_commands
from discord.ext import commands, tasks
from discord_webhook import DiscordWebhook

from github import Auth, Github
from dotenv import load_dotenv

import functions
import yfinance as yf
from themes import brand, bgDark

load_dotenv()
WEBHOOK = os.getenv("FEEDBACK_WEBHOOK")
GIT = os.getenv("GIT_TOKEN")

# map of discord user id -> asyncio.Future used to await registration results
REGISTRATIONS: dict[int, asyncio.Future] = {}
MODELS = ["Implied Volatility", "Extrapolation", "Grounded-Extrapolation", "Logical Analysis [UNAVAILABLE]"]
BOT_INVITE = "IN BETA"
DEFAULT_ICON = "bot/assets/icons/discord.png"

HUMANIZER = functions.Humanizer()
gitClient = Github(auth=Auth.Token(GIT))

VERIFY_CHANNEL_ID = int(os.getenv("VERIFY_CHANNEL_ID", "1482515397138972885"))
ACCESS_ROLE_ID = int(os.getenv("ACCESS_ROLE_ID", "1482476984159436800"))

def getVersion():
    RELEASE = 2
    try:
        user = gitClient.get_user()
        commits = user.get_repo("RICH").get_commits().totalCount
        return RELEASE + commits/1000
    except:
        return RELEASE

def getStatic(info):
    return {
        "getDayOpen": info.getDayOpen(),
        "getDayClose": info.getDayClose(),
        "get52wkHigh": info.get52wkHigh(),
        "get52wkLow": info.get52wkLow(),
        "getVolume": info.getVolume(),
        "getAvgVolume": info.getAvgVolume(),
        "getPERatio": info.getPERatio(),
        "getEPSRatio": info.getEPSRatio(),
        "getBeta": info.getBeta(),
        "getMktCap": info.getMktCap(),
        "getAnnualYield": info.getAnnualYield(),
        "getMonthlyYield": info.getMonthlyYield(),
        "getExDividendDate": info.getExDividendDate(),
        "getPayDate": info.getPayDate(),
        "getDividendAmount": info.getDividendAmount(),
        "getDividendChange": info.getDividendChange(),
    }

def infoEmbed(info: any, ticker: str, static: dict):
    embed = discord.Embed(
        color=discord.Colour.teal(), 
        title=f"{round(info.getCurrentPrice(),2)} ({HUMANIZER.sign(round(info.getPriceChange(),2))}%)"
    )
    embed.set_author(name=f"{str.upper(ticker)}")
    embed.add_field(name=f"Open: {static.get('getDayOpen'):.2f}", value=f"Close*: {static.get('getDayClose'):.2f}", inline=True)
    embed.add_field(name=f"High: {info.getDayHigh():.2f}", value=f"Low: {info.getDayLow():.2f}", inline=True)
    embed.add_field(name=f"52W H: {static.get('get52wkHigh'):.2f}", value=f"52W L: {static.get('get52wkLow'):.2f}", inline=True)

    embed.add_field(name=f"Volume: {HUMANIZER.suffix(static.get('getVolume'))}", value=f"Avg Volume: {HUMANIZER.suffix(static.get('getAvgVolume'))}", inline=True)
    embed.add_field(name=f"P/E: {static.get('getPERatio'):.2f}", value=f"EPS: {static.get('getEPSRatio'):.2f}", inline=True)
    embed.add_field(name=f"Beta: {static.get('getBeta'):.2f}", value=f"Mkt Cap: {HUMANIZER.suffix(static.get('getMktCap'))}", inline=True)

    embed.add_field(name=f"Annual Yield: {static.get('getAnnualYield')}%", value=f"Monthly Yield: {static.get('getMonthlyYield')}%", inline=True)
    embed.add_field(name=f"Ex. Div.: {static.get('getExDividendDate')}", value=f"Div. Payout: {static.get('getPayDate')}")
    embed.add_field(name=f"Expected Amount: {static.get('getDividendAmount')}", value=f"Expected Change: {static.get('getDividendChange')}")
    embed.set_footer(text="* is previous day's close, with the exception of aftermarket, whereby 'Close' is the day's close")
    return embed

class Robot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.checkAlerts.start()
        self.trainPopular.start()

    def cogUnload(self):
        self.checkAlerts.cancel()
        self.trainPopular.cancel()

    @tasks.loop(minutes=1)
    async def checkAlerts(self):
        try:
            alerts = functions.getAllAlerts()
            if not alerts: return
            
            symbols = list(set(a[2] for a in alerts))
            prices = {}
            for sym in symbols:
                try:
                    s = yf.Ticker(sym)
                    prices[sym] = s.fast_info.get("lastPrice")
                except Exception: pass
            
            for alertId, discordId, symbol, targetPrice in alerts:
                currentPrice = prices.get(symbol)
                if currentPrice:
                    if abs(currentPrice - targetPrice) / targetPrice < 0.005:
                        try:
                            user = await self.bot.fetch_user(int(discordId))
                            if user:
                                embed = discord.Embed(color=discord.Colour.teal(), title=f"{symbol} Price Alert")
                                embed.description = f"**{symbol}** has reached your target of **${targetPrice:.2f}**!"
                                embed.set_footer(text="Alert deleted. Use /alerts to create a new one.")
                                await user.send(embed=embed)
                                functions.removeAlert(alertId)
                        except Exception: pass
        except Exception:
            traceback.print_exc()

    @tasks.loop(hours=24)
    async def trainPopular(self):
        now = datetime.datetime.now()
        if now.weekday() != 5:
            return
            
        try:
            with functions.DB_LOCK:
                with functions.DB_CONNECTION.cursor() as cursor:
                    window = int((datetime.datetime.now() - datetime.timedelta(days=14)).timestamp())
                    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM Request WHERE CAST(updated AS BIGINT) >= %s", (window,))
                    row = cursor.fetchone()
                    if not row or row[0] == 0: return
                    totalSymbols = row[0]
                    limit = max(1, int(totalSymbols * 0.10))
                    
                    cursor.execute("""
                        SELECT t.ticker, COUNT(r.id) as reqCount 
                        FROM Request r 
                        JOIN Ticker t ON t.id = r.ticker 
                        WHERE CAST(r.updated AS BIGINT) >= %s
                        GROUP BY t.ticker 
                        ORDER BY reqCount DESC 
                        LIMIT %s
                    """, (window, limit))
                    rows = cursor.fetchall()
            
            tickersToTrain = [r[0] for r in rows if r[0]]
            
            # Ensure sector ETFs and Macro indexes are always trained during the weekend batch
            sector_etfs = list(functions.Charts._SECTOR_MAP.values())
            macro_indexes = ["^GSPC"]
            for etf in sector_etfs + macro_indexes:
                if etf not in tickersToTrain:
                    tickersToTrain.append(etf)
            
            if tickersToTrain:
                charts = functions.Charts()
                for ticker in tickersToTrain:
                    try:
                        await asyncio.to_thread(charts._liveTrain, ticker)
                    except Exception:
                        pass
        except Exception:
            traceback.print_exc()

    @staticmethod
    def _registered(discordID):
        user = functions.User(discordID=discordID)
        if user.accountFromDiscord() is not None: return True
        return False

    def lookup(self, query, header="Results for", boolean=False):
        results = yf.Lookup(query).get_all(10)
        symbols = results.index.to_list() if "exchange" not in results.columns else results["exchange"].index.to_list()
        names = results["shortName"] if "shortName" in results else None
        sanity = query.upper() in [s.upper() for s in symbols]
        if boolean: return sanity
        if results is None or results.empty or results.index.empty: 
            embed = discord.Embed(color=discord.Colour.teal(), title="404: Not Found")
            embed.description = f"No suggestions found for **{query}**. Please check your spelling."
            return embed

        desc = ""
        embed = discord.Embed(color=discord.Colour.teal(),title=f"{header.strip()} {query.upper()}:")

        for i, name in enumerate(names):
            symbol = str(symbols[i])
            if names is not None: name = str(names.iloc[i]) if names.iloc[i] is not None else "Unknown"
            else: name = "Unknown"
            name = re.sub(" +"," ",name)
            desc += f"- ({symbol}) {name}\n"

        embed.description = desc
        return embed
    
    async def authenticated(self, interaction: discord.Interaction, bypass=False):
        if self._registered(interaction.user.id) and bypass==False:
            return True

        embed = discord.Embed(color=discord.Colour.teal(), title="401: Unauthorized")
        embed.description = "## Getting Started\n**You must have a registered account before accessing our services.**\nPlease take a few minutes to read our Terms of Service and Privacy Policy.\nYou can review these documents using the buttons below.\n\nMake a selection from the two menus presented:**\n1. Confirm marketing communication preferences.\n2. Confirm that you have read, understand, and agree to our EULA.**"
        
        try:
            if interaction.response.is_done(): await interaction.followup.send(embed=embed, view=RegisterPrompt(interaction.user.id), ephemeral=True)
            else: await interaction.response.send_message(embed=embed, view=RegisterPrompt(interaction.user.id), ephemeral=True)
        except Exception:
            try: await interaction.followup.send(embed=embed, view=RegisterPrompt(interaction.user.id), ephemeral=True)
            except Exception: pass

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        REGISTRATIONS[interaction.user.id] = fut

        try:
            result = await asyncio.wait_for(fut, timeout=300)
            return bool(result)
        except asyncio.TimeoutError:
            REGISTRATIONS.pop(interaction.user.id, None)
            timeoutEmbed = discord.Embed(color=discord.Colour.teal(), title="408: Registration Timeout")
            timeoutEmbed.description = "Registration timed out. Please try again."
            await interaction.followup.send(embed=timeoutEmbed, ephemeral=True)
            return False

    async def on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        traceback.print_exc()
        embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
        embed.description = "An unexpected error occurred. Please try again later."
        try:
            if interaction.response.is_done(): await interaction.followup.send(embed=embed, ephemeral=True)
            else: await interaction.response.send_message(embed=embed, ephemeral=True)
        except Exception: pass

    @app_commands.command(name="help", description="List all commands, and additional information")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def help(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            embed = discord.Embed(color=discord.Colour.teal(), title=f"Quantum (v{getVersion()})")
            with open("README.md", "r", encoding="utf-8") as md:
                content = md.read()
                usage_start = content.find("## Usage")
                embed.description = content[usage_start:] if usage_start != -1 else "Usage documentation not found."
            await interaction.followup.send(embed=embed)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Sorry, an error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="quote", description="Provide latest quote of a given ticker only")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)")
    async def quote(self, interaction: discord.Interaction, ticker: str):
        try:
            await interaction.response.defer()
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            sanity = self.lookup(ticker,boolean=True)
            if sanity == False:
                await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return

            functions.recordRequest(interaction.user.id, ticker)
            info = functions.yFinanceWrapper(ticker=ticker)
            static = getStatic(info)
            
            update = Update(ticker=ticker)
            embed = infoEmbed(info=info, ticker=ticker, static=static) if sanity else None
            await interaction.followup.send(embed=embed, view=update)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Sorry, an error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="chart", description="Provide latest chart and quote of a given ticker")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)", duration="Time range of data to display on the graph", interval="How much to zoom in for the range of data")
    @app_commands.choices(duration=[
        app_commands.Choice(name="Past 24 Hours (1d)", value="1d"),
        app_commands.Choice(name="Past Week (5d)", value="5d"),
        app_commands.Choice(name="Past Month (1mo)", value="1mo"),
        app_commands.Choice(name="Past 3 Months (3mo)", value="3mo"),
        app_commands.Choice(name="Past 6 Months (6mo)", value="6mo"),
        app_commands.Choice(name="Past Year (1y)", value="1y"),
        app_commands.Choice(name="Past Year from Today (ytd)", value="ytd"),
        app_commands.Choice(name="Past 2 Years (2y)", value="2y"),
        app_commands.Choice(name="Past 5 Years (5y)", value="5y"),
        app_commands.Choice(name="Past 10 Years (10y)", value="10y"),
        app_commands.Choice(name="Maximum displayable (all)", value="max"),
    ])
    @app_commands.choices(interval=[
        app_commands.Choice(name="2 Minutes Between", value="2m"),
        app_commands.Choice(name="15 Minutes Between", value="15m"),
        app_commands.Choice(name="30 Minutes Between", value="30m"),
        app_commands.Choice(name="1 Hour Between", value="60m"),
        app_commands.Choice(name="1 Day Between", value="1d"),
        app_commands.Choice(name="1 Week (5d) Between", value="5d"),
        app_commands.Choice(name="1 Month Between", value="1mo"),
        app_commands.Choice(name="3 Months Between", value="3mo"), #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    ])
    async def chart(self, interaction: discord.Interaction, ticker: str, duration:str, interval: typing.Optional[app_commands.Choice[str]]):
        try:
            await interaction.response.defer(ephemeral=True)
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            sanity = self.lookup(ticker,boolean=True)
            if sanity == False:
                await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return
            
            charts = functions.Charts()

            info = functions.yFinanceWrapper(ticker=ticker)
            static = getStatic(info)

            update = Update(ticker=ticker)
            embed = infoEmbed(info=info, ticker=ticker, static=static) if sanity else None
            
            if interaction.guild:
                invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
                icon = interaction.guild.icon.url if interaction.guild.icon else DEFAULT_ICON
                serverName = interaction.guild.name
                inviteUrl = invite.url
            else:
                icon = DEFAULT_ICON
                serverName = "QuantumBot"
                inviteUrl = BOT_INVITE

            loading = discord.Embed(color=discord.Colour.teal(), title="Generating Chart...")
            loading.description = "Starting..."
            status = await interaction.followup.send(embed=loading)

            import time
            import random
            startTime = time.time()
            task = asyncio.create_task(asyncio.to_thread(
                charts.history, 
                ticker, 
                duration, 
                interval.value if interval else None, 
                serverName, 
                inviteUrl, 
                icon, 
                static,
                interaction.user.id
            ))

            async def loading_loop():
                assets = [f for f in os.listdir("bot/assets/marketing") if f.endswith(('.png', '.jpg', '.jpeg'))]
                while not task.done():
                    statusText = functions.STATUS_REGISTRY.get(interaction.user.id, "Processing...")
                    try:
                        e = discord.Embed(color=discord.Colour.teal(), title="Generating Chart...")
                        e.description = statusText
                        if assets:
                            chosen = random.choice(assets)
                            banner_file = discord.File(f"bot/assets/marketing/{chosen}", filename="banner.png")
                            e.set_image(url="attachment://banner.png")
                            await status.edit(embed=e, attachments=[banner_file])
                        else:
                            await status.edit(embed=e)
                    except Exception: pass
                    await asyncio.sleep(5)

            loopTask = asyncio.create_task(loading_loop())
            img = await task
            loopTask.cancel()
            functions.STATUS_REGISTRY.pop(interaction.user.id, None)

            elapsed = time.time() - startTime
            if elapsed < 5:
                await asyncio.sleep(5 - elapsed)
            if img:
                file = discord.File(img, filename="output.png")
                embed.set_image(url="attachment://output.png")
                await interaction.followup.send(file=file, embed=embed, view=update, ephemeral=False)
                await status.delete()
        except AssertionError as e:
            embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
            embed.description = "Please check your intervals and duration again."
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Sorry, an error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="alerts", description="Create/check/clear alerts")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def alerts(self, interaction: discord.Interaction):
        try:
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            view = AlertsMenu(self.bot)
            await interaction.response.send_message("## Alerts Menu\nChoose an action below to manage your ticker alerts.", view=view, ephemeral=True)
        except Exception:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Sorry, an error occurred while opening the alerts menu."
            await interaction.response.send_message(embed=embed, ephemeral=True)
    
    @app_commands.command(name="predict", description="Predicts future movements of a given ticker")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(ticker="The ticker symbol to predict (ex. AAPL)", model="Choose model algorithm", lookback="Historical data window for analysis")
    @app_commands.choices(model=[
        app_commands.Choice(name=MODELS[0], value="0"),
        app_commands.Choice(name=MODELS[1], value="1"),
        app_commands.Choice(name=MODELS[2], value="2"),
        ])
    @app_commands.choices(lookback=[
        app_commands.Choice(name="Past 7 Days", value="7d"),
        app_commands.Choice(name="Past 14 Days", value="14d"),
        app_commands.Choice(name="Past Month (30d)", value="30d"),
        app_commands.Choice(name="Past 3 Months (90d)", value="90d"),
        app_commands.Choice(name="Past 6 Months (180d)", value="180d"),
        app_commands.Choice(name="Past Year from Date (ytd)", value="ytd"),
        ])
    async def predict(self, interaction: discord.Interaction, ticker: str, model: typing.Optional[app_commands.Choice[str]], lookback: typing.Optional[app_commands.Choice[str]]):
        try:
            await interaction.response.defer(ephemeral=True)
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            if self.lookup(ticker,boolean=True) == False:
                await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return
            
            charts = functions.Charts()
            selectedLookback = lookback.value if lookback else "90d"

            embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} Prediction ({selectedLookback})")
            embed.set_footer(text="Source: Yahoo Finance @ %s"%(datetime.datetime.now().strftime("%m/%d/%Y @ %H:%M:%S")))
            
            selectedModel = int(model.value) if model else 2
            warning = False

            if interaction.guild:
                invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
                icon = interaction.guild.icon.url if interaction.guild.icon else DEFAULT_ICON
                serverName = interaction.guild.name
                inviteUrl = invite.url
            else:
                icon = DEFAULT_ICON
                serverName = "QuantumBot"
                inviteUrl = BOT_INVITE

            loading = discord.Embed(color=discord.Colour.teal(), title="Generating Prediction...")
            loading.description = "Starting Thread..."
            status = await interaction.followup.send(embed=loading)

            startTime = time.time()

            async def loadingLoop():
                assets = [f for f in os.listdir("bot/assets/marketing") if f.endswith(('.png', '.jpg', '.jpeg'))]
                while not task.done():
                    statusText = functions.STATUS_REGISTRY.get(interaction.user.id, "Processing...")
                    try:
                        e = discord.Embed(color=discord.Colour.teal(), title="Generating Prediction...")
                        e.description = statusText
                        if assets:
                            chosen = random.choice(assets)
                            bannerFile = discord.File(f"bot/assets/marketing/{chosen}", filename="banner.png")
                            e.set_image(url="attachment://banner.png")
                            await status.edit(embed=e, attachments=[bannerFile])
                        else:
                            await status.edit(embed=e)
                    except Exception: pass
                    await asyncio.sleep(5)

            task = asyncio.create_task(asyncio.to_thread(charts.project, ticker, selectedModel, serverName, inviteUrl, icon, interaction.user.id, selectedLookback))
            loopTask = asyncio.create_task(loadingLoop())
            img, (predictedPrice, weights) = await task
            
            if img is None:
                warning = True
                functions.STATUS_REGISTRY[interaction.user.id] = "Using Extrapolation Model..."
                task = asyncio.create_task(asyncio.to_thread(charts.project, ticker, 1, serverName, inviteUrl, icon, interaction.user.id, selectedLookback))
                img, (predictedPrice, weights) = await task

            loopTask.cancel()
            functions.STATUS_REGISTRY.pop(interaction.user.id, None)

            elapsed = time.time() - startTime
            if elapsed < 5:
                await asyncio.sleep(5 - elapsed)

            if img:
                functions.STATUS_REGISTRY[interaction.user.id] = "Finalizing/Cleaning..."
                imgCopy = io.BytesIO(img.getvalue())
                file = discord.File(imgCopy, filename="output.png")
                embed.set_image(url="attachment://output.png")

                feedbackView = Feedback(predictedPrice, ticker, selectedModel, imgCopy, serverName, inviteUrl, icon, selectedLookback, currentWeights=weights)
                if warning: embed.description = "WARNING: Model has been changed because there were not enough datapoints to draw an accurate conclusion."

                await interaction.followup.send(file=file, embed=embed, view=feedbackView)
                await status.delete()
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Sorry, an error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="tickers", description="Check/find the exact ticker for a given query (stock, index, etf, general search)")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(query="The input to validate")
    async def tickers(self, interaction: discord.Interaction, query:str):
        try:
            await interaction.response.defer()
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            await interaction.followup.send(embed=self.lookup(query=query))
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Sorry, an error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @commands.command(name="evaluate")
    async def evaluate(self, ctx, *, ticker_input: str):
        # Split by comma or space
        tickers = [t.strip().upper() for t in re.split(r'[,\s]+', ticker_input) if t.strip()]
        
        if not tickers:
            await ctx.send("Please provide at least one ticker symbol.")
            return

        status_msg = await ctx.send(f"Batch evaluating: **{', '.join(tickers)}**... (This may take a moment)")
        
        charts = functions.Charts()
        results_processed = []
        llm_report = ["### BACKTEST EVALUATION REPORT (90-DAY)"]

        for ticker in tickers:
            try:
                # Update status for current ticker
                await status_msg.edit(content=f"Evaluating **{ticker}**... ({len(results_processed)+1}/{len(tickers)})")
                
                result = await asyncio.to_thread(charts.evaluate, ticker)
                
                if result:
                    buf, diffs, (low, high) = result
                    file = discord.File(buf, filename=f"evaluation_{ticker}.png")
                    await ctx.send(f"**{ticker} Visual Results**", file=file)
                    
                    # Format divergence data for the report
                    diff_text = ", ".join([f"{d:+.2f}" for d in diffs])
                    llm_report.append(f"\n{ticker.upper()}:")
                    llm_report.append(f"Price Range: ${low:.2f} - ${high:.2f}")
                    llm_report.append(f"Divergence: {diff_text}")
                    
                    results_processed.append(ticker)
                else:
                    await ctx.send(f"⚠️ Error: Could not evaluate **{ticker}**.")
            except Exception:
                traceback.print_exc()
                await ctx.send(f"❌ Critical failure evaluating **{ticker}**.")

        # Final Report Delivery
        if results_processed:
            full_report = "```\n" + "\n".join(llm_report) + "\n```"
            # Handle Discord 2000 char limit
            if len(full_report) > 1950:
                chunks = [full_report[i:i+1950] for i in range(0, len(full_report), 1950)]
                for chunk in chunks:
                    await ctx.send(f"```\n{chunk.replace('```','')}\n```")
            else:
                await ctx.send(full_report)

        await status_msg.edit(content=f"✅ Evaluation complete for: **{', '.join(results_processed)}**")

    @app_commands.command(name="me", description="Display account information")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def me(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        try:
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            user = functions.User(discordID=interaction.user.id)
            analytics = user.getAnalytics()
            
            embed = discord.Embed(color=discord.Colour.teal(), title=f"Account Information")
            desc = f"Discord ID: {interaction.user.id}\nUsername: {interaction.user.name}\nRegistered: Yes\n\n"
            desc += f"**Requests Made:**\n"
            desc += f"Total: {analytics['total']}\n"
            desc += f"Monthly: {analytics['monthly']}\n"
            desc += f"Weekly: {analytics['weekly']}\n"
            desc += f"Daily: {analytics['daily']}\n"
            
            embed.description = desc
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, an error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

class ServerAccess(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Verify", style=discord.ButtonStyle.green, custom_id="verify_server_access")
    async def verify_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        user = functions.User(interaction.user.id)
        if user.accountFromDiscord() is not None:
            # Assign role
            role = interaction.guild.get_role(ACCESS_ROLE_ID)
            if role:
                try:
                    await interaction.user.add_roles(role)
                    embed = discord.Embed(color=discord.Colour.teal(), title="Verification Successful")
                    embed.description = "## Welcome Back\nYou already have a registered account! You have been granted access to channels."
                    await interaction.response.send_message(embed=embed, ephemeral=True)
                except Exception as e:
                    traceback.print_exc()
                    embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
                    embed.description = "Sorry, an error occurred while assigning your role. Please contact support if the issue persists."
                    await interaction.response.send_message(embed=embed, ephemeral=True)
            else:
                embed = discord.Embed(color=discord.Colour.teal(), title="404: Not Found")
                embed.description = "Critical Error: The access role was not found in this guild. Please contact the server administrator."
                await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        embed = discord.Embed(color=discord.Colour.teal(), title="401: Unauthorized")
        embed.description = "## Getting Started\n**You must have a registered account before accessing our services.**\nPlease take a few minutes to read our Terms of Service and Privacy Policy.\nYou can review these documents using the buttons below.\n\nMake a selection from the two menus presented:**\n1. Confirm marketing communication preferences.\n2. Confirm that you have read, understand, and agree to our EULA.**"
        
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        REGISTRATIONS[interaction.user.id] = fut
        
        await interaction.response.send_message(embed=embed, view=RegisterPrompt(interaction.user.id), ephemeral=True)

        try:
            await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            REGISTRATIONS.pop(interaction.user.id, None)
            try:
                timeoutEmbed = discord.Embed(color=discord.Colour.teal(), title="408: Registration Timeout")
                timeoutEmbed.description = "Registration timed out. Please try again."
                await interaction.followup.send(embed=timeoutEmbed, ephemeral=True)
            except Exception: pass

class RegisterPrompt(discord.ui.View):
    def __init__(self, user_id):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.marketing: typing.Optional[bool] = None
        self.legal: typing.Optional[bool] = None
        self.add_item(discord.ui.Button(label="Terms of Service", url="https://github.com/razoring/QuantumDiscordBot/blob/main/TermsService", row=0))
        self.add_item(discord.ui.Button(label="Privacy Policy", url="https://github.com/razoring/QuantumDiscordBot/blob/main/PrivacyPolicy", row=0))

    @discord.ui.select(
        placeholder="Confirm your marketing preferences.",
        options=[
            discord.SelectOption(label="AGREE", value="True", description="I AGREE to receive marketing communications regarding new features, promotions, and more."),
            discord.SelectOption(label="DISAGREE", value="False", description="I DISAGREE to receive marketing communications regarding new features, promotions, and more.")
        ],
        row=1,
        custom_id="register_marketing"
    )
    async def select_marketing(self, interaction: discord.Interaction, select: discord.ui.Select):
        if interaction.user.id != self.user_id:
            embed = discord.Embed(color=discord.Colour.teal(), title="403: Forbidden")
            embed.description = "This menu is not for you."
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.marketing = select.values[0] == "True"
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="Confirm you read and understand the EULA.",
        options=[
            discord.SelectOption(label="AGREE", value="True", description="I have read, understand, and AGREE to the legal disclaimers, terms, conditions, and privacy policy."),
            discord.SelectOption(label="DISAGREE", value="False", description="I have read, and DISAGREE to the terms. Thereby I confirm that I cannot use bot's features.")
        ],
        row=2,
        custom_id="register_legal"
    )
    async def select_legal(self, interaction: discord.Interaction, select: discord.ui.Select):
        if interaction.user.id != self.user_id:
            embed = discord.Embed(color=discord.Colour.teal(), title="403: Forbidden")
            embed.description = "This menu is not for you."
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.legal = select.values[0] == "True"
        await interaction.response.defer()

    @discord.ui.button(label="Complete Registration", style=discord.ButtonStyle.green, row=3, custom_id="register_submit")
    async def register_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            embed = discord.Embed(color=discord.Colour.teal(), title="403: Forbidden")
            embed.description = "This menu is not for you."
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
            
        if self.marketing is None or self.legal is None:
            embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
            embed.description = "You must make both selections."
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        fut = REGISTRATIONS.pop(self.user_id, None)
        
        if not self.legal:
            if fut and not fut.done(): fut.set_result(False)
            embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
            embed.description = "Registration declined. You must agree to the terms to use QuantumBot."
            await interaction.response.edit_message(embed=embed, view=None)
            return

        user = functions.User(self.user_id)
        if user.createAccount(marketing=self.marketing):
            # Assign role
            if interaction.guild:
                role = interaction.guild.get_role(ACCESS_ROLE_ID)
                if role:
                    try:
                        await interaction.user.add_roles(role)
                    except Exception: pass

            if fut and not fut.done(): fut.set_result(True)
            embed = discord.Embed(color=discord.Colour.teal(), title="Account Creation Successful")
            embed.description = "## Welcome to QuantumBot\nYour account has been successfully created. You have been granted full access to our forecasting tools and private channels."
            await interaction.response.edit_message(embed=embed, view=None)
        else:
            if fut and not fut.done(): fut.set_result(False)
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "Critical Error: Registration failed. Please contact support."
            await interaction.response.send_message(embed=embed, ephemeral=True)

class Update(discord.ui.View):
    def __init__(self, ticker):
        super().__init__(timeout=300)
        self.ticker = ticker
    
    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.gray, custom_id="Refresh")
    async def refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
        functions.recordRequest(interaction.user.id, self.ticker)
        new_info = functions.yFinanceWrapper(ticker=self.ticker)
        new_static = getStatic(new_info)
        
        await interaction.response.edit_message(embed=infoEmbed(info=new_info, ticker=self.ticker, static=new_static), view=self)

class Feedback(discord.ui.View):
    def __init__(self, alertPrice, alertTicker, model, fileObject, serverName, serverInvite, serverIcon, lookback="90d", currentWeights=None):
        super().__init__(timeout=300)
        self.alertPrice = alertPrice
        self.ticker = alertTicker
        self.model = model
        self.file = fileObject
        self.serverName = serverName
        self.serverInvite = serverInvite
        self.serverIcon = serverIcon
        self.lookback = lookback
        self.currentWeights = currentWeights
        self._processing = False
        self._updateLabels()

    def _updateLabels(self):
        likes, dislikes = functions.getTickerFeedback(self.ticker)
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                if child.custom_id == "LikeButton":
                    child.label = f"👍 {likes}" if likes > 0 else "👍"
                elif child.custom_id == "DislikeButton":
                    child.label = f"👎 {dislikes}" if dislikes > 0 else "👎"

    async def feedbackSubmit(self, interaction: discord.Interaction, rating):
        if self._processing: return
        self._processing = True
        try:
            confirmEmbed = discord.Embed(color=discord.Color.teal(), title="Vote Received")
            voteType = "negative" if rating == "👎" else "positive"
            confirmEmbed.description = f"Your **{voteType}** feedback for **{self.ticker.upper()}** has been recorded."
            if rating == "👎":
                confirmEmbed.description += " The prediction is being recalculated..."
            
            await interaction.response.send_message(embed=confirmEmbed, ephemeral=True)
            originalMessage = interaction.message

            await asyncio.to_thread(functions.recordPredictionFeedback, self.ticker, rating, self.currentWeights)
            
            if rating == "👎":
                if originalMessage:
                    await originalMessage.edit(view=None)

                # Mutate weights locally for the session
                mutatedWeights = await asyncio.to_thread(functions.mutateWeights, self.ticker, self.currentWeights)

                charts = functions.Charts()
                img, (predictedPrice, newWeights) = await asyncio.to_thread(
                    charts.project, 
                    self.ticker, 
                    self.model, 
                    self.serverName, 
                    self.serverInvite, 
                    self.serverIcon,
                    interaction.user.id,
                    self.lookback,
                    overriddenWeights=mutatedWeights
                )
                
                if img:
                    self.file = io.BytesIO(img.getvalue())
                    self.alertPrice = predictedPrice
                    self.currentWeights = newWeights

                self._updateLabels()
                
                if originalMessage:
                    attempts = 0
                    success = False
                    while attempts < 3 and not success:
                        try:
                            attempts += 1
                            self.file.seek(0)
                            file = discord.File(self.file, filename="output.png")
                            if len(originalMessage.embeds) > 0:
                                embed = originalMessage.embeds[0]
                                embed.set_image(url="attachment://output.png")
                                await originalMessage.edit(attachments=[file], embed=embed, view=self)
                            else:
                                await originalMessage.edit(attachments=[file], view=self)
                            success = True
                        except Exception as e:
                            if attempts >= 3: raise e
                            await asyncio.sleep(1)
            
            else: # rating == "👍"
                self._updateLabels()
                if originalMessage:
                    attempts = 0
                    success = False
                    while attempts < 3 and not success:
                        try:
                            attempts += 1
                            await originalMessage.edit(view=self)
                            success = True
                        except Exception as e:
                            if attempts >= 3: raise e
                            await asyncio.sleep(1)

        except Exception:
            traceback.print_exc()
            try:
                errorEmbed = discord.Embed(color=discord.Color.red(), title="Error", description="An error occurred while processing your feedback. Please try again.")
                await interaction.followup.send(embed=errorEmbed, ephemeral=True)
            except: pass
        finally:
            self._processing = False

    @discord.ui.button(label="Set Alert", style=discord.ButtonStyle.green, custom_id="SetAlertButton")
    async def setAlert(self, interaction: discord.Interaction, button: discord.ui.Button):
        user = functions.User(interaction.user.id)
        if user.createAlert(self.ticker, self.alertPrice):
            embed = discord.Embed(color=discord.Colour.teal(), title="Alert Set")
            embed.description = f"Success: Alert set for **{self.ticker.upper()}** at **${self.alertPrice:.2f}**"
            await interaction.response.send_message(embed=embed, ephemeral=True)
            self.remove_item(button)
            await interaction.edit_original_response(view=self)
        else:
            embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
            embed.description = "Error: Could not set alert. Please ensure you have an account (/register)."
            await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(label="👍", style=discord.ButtonStyle.gray, custom_id="LikeButton")
    async def likeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.feedbackSubmit(interaction, "👍")

    @discord.ui.button(label="👎", style=discord.ButtonStyle.gray, custom_id="DislikeButton")
    async def dislikeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.feedbackSubmit(interaction, "👎")

    @discord.ui.button(label="Feedback", style=discord.ButtonStyle.gray, custom_id="DetailedFeedbackButton")
    async def detailsButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(FeedbackModal(self.ticker, self.model, self.file))

class FeedbackModal(discord.ui.Modal):
    def __init__(self, ticker, model, fileObject):
        super().__init__(title="Feedback", custom_id="modal")
        self.ticker = ticker
        self.model = model
        self.file = fileObject

        self.fileUpload = discord.ui.TextInput(
            label="File Upload",
            placeholder="Link to photo or screenshot...",
            style=discord.TextStyle.short,
            custom_id="4f0e86a466214dff9d1558d15809b696",
            required=False
        )
        
        self.descriptionInput = discord.ui.TextInput(
            label="Description",
            placeholder="Explain the issue, bug or inaccuracy with great detail. What caused it and how do you replicate it?",
            style=discord.TextStyle.paragraph,
            custom_id="5582e68567f84a29b9aff56131669417",
            min_length=50,
            required=True
        )
        
        self.add_item(self.fileUpload)
        self.add_item(self.descriptionInput)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        self.file.seek(0)
        
        content = (
            f"**Detailed Feedback Received**\n"
            f"Ticker: {self.ticker}\n"
            f"Model: {self.model}\n"
            f"Version: {getVersion()}\n"
            f"Description: {self.descriptionInput.value}\n"
            f"Attached Photo Link: {self.fileUpload.value if self.fileUpload.value else 'None'}"
        )
        
        hook = DiscordWebhook(url=WEBHOOK, content=content)
        hook.add_file(file=self.file.getvalue(), filename="prediction_chart.png")
        await asyncio.to_thread(hook.execute)
        
        success_embed = discord.Embed(color=discord.Colour.teal(), title="Feedback Submitted")
        success_embed.description = "Thank you for your detailed feedback! Our team will review it."
        await interaction.followup.send(embed=success_embed, ephemeral=True)

class AlertsDropdown(discord.ui.Select):
    def __init__(self, bot):
        options = [
            discord.SelectOption(label="Create Ticker Alert", value="20a81ebd5d074c38b7c7bbade008d082", description="Create a new alert at a specific price. You will enter the required information in a further menu."),
            # discord.SelectOption(label="Create Volatility Alert", value="55966df684dc4405b04306ee462db149", description="Daily/weekly/monthly closing alerts."),
            # discord.SelectOption(label="Create Market Hours Alert", value="90c3f353d7ba4f3296667dce62df6768", description="Market open/close notifications."),
            discord.SelectOption(label="List Alerts", value="7e1f307aec2a4e63b04eefe00402faa6", description="Display all active ticker alerts. No further action necessary."),
            discord.SelectOption(label="Clear Alerts", value="ca0d8fe917d649039e9f9bc5e20c0fb2", description="Clear all alerts. This action cannot be undone.")
        ]
        super().__init__(placeholder="Choose action", min_values=1, max_values=1, options=options, custom_id="b11fb05bbc12470ab2e9a76f3d1290a9")
        self.bot = bot

    async def callback(self, interaction: discord.Interaction):
        try:
            if self.values[0] == "20a81ebd5d074c38b7c7bbade008d082":
                await interaction.response.send_modal(AlertCreateModal(self.bot))
            elif self.values[0] == "7e1f307aec2a4e63b04eefe00402faa6":
                await interaction.response.defer(ephemeral=True)
                user = functions.User(interaction.user.id)
                alerts = user.getAlerts()
                if not alerts:
                    embed = discord.Embed(color=discord.Colour.teal(), title="No Alerts Found")
                    embed.description = "You have no active alerts."
                    await interaction.followup.send(embed=embed, ephemeral=True)
                else:
                    embed = discord.Embed(title="Your Active Alerts", color=discord.Color.teal())
                    desc = ""
                    for sym, price, updated in alerts:
                        desc += f"• **{sym}**: ${price:.2f} (Set: <t:{updated}:R>)\n"
                    embed.description = desc
                    await interaction.followup.send(embed=embed, ephemeral=True)
            elif self.values[0] == "ca0d8fe917d649039e9f9bc5e20c0fb2":
                await interaction.response.defer(ephemeral=True)
                user = functions.User(interaction.user.id)
                if user.clearAlerts():
                    embed = discord.Embed(color=discord.Colour.teal(), title="Alerts Cleared")
                    embed.description = "Success: All alerts have been cleared."
                    await interaction.followup.send(embed=embed, ephemeral=True)
                else:
                    embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
                    embed.description = "Error: Failed to clear alerts. Please ensure you have an account (!me)."
                    await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception:
            traceback.print_exc()

class AlertsMenu(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=180)
        self.add_item(AlertsDropdown(bot))

class AlertCreateModal(discord.ui.Modal, title="Create Ticker Alert"):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.ticker = discord.ui.TextInput(
            label="Ticker", 
            placeholder="ex. AAPL", 
            style=discord.TextStyle.short, 
            custom_id="529799a9fe0b4d0c83a20fae114c5066",
            required=True
        )
        self.price = discord.ui.TextInput(
            label="Price", 
            placeholder="149.99", 
            style=discord.TextStyle.short, 
            custom_id="76be0e8530124b39b508d9e44724b3d7",
            required=True
        )
        self.add_item(self.ticker)
        self.add_item(self.price)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            ticker_val = self.ticker.value.strip().upper()
            price_str = self.price.value.strip().replace("$", "")
            
            try:
                price_val = float(price_str)
            except ValueError:
                embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
                embed.description = "Error: Invalid price format. Please enter a positive number."
                await interaction.followup.send(embed=embed, ephemeral=True)
                return

            user = functions.User(interaction.user.id)
            if user.createAlert(ticker_val, price_val):
                embed = discord.Embed(color=discord.Colour.teal(), title="Alert Created")
                embed.description = f"Success! You will be notify in DMs when **{ticker_val}** reaches **${price_val:.2f}**."
                await interaction.followup.send(embed=embed, ephemeral=True)
            else:
                embed = discord.Embed(color=discord.Colour.teal(), title="400: Bad Request")
                embed.description = "Error: Failed to create alert. Please initialize your account with `/me` or try again."
                await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Internal Server Error")
            embed.description = "An unexpected error occurred while creating the alert."
            await interaction.followup.send(embed=embed, ephemeral=True)

# mount bot - DO NOT TOUCH
async def setup(bot): await bot.add_cog(Robot(bot))