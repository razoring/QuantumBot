import io
import os
import re
import traceback
import typing
import datetime
import asyncio
from urllib.parse import urlparse as url

import discord
from discord import app_commands
from discord.ext import commands
from discord_webhook import DiscordWebhook

from github import Auth, Github
from dotenv import load_dotenv

import functions
import yfinance as yf
from themes import brand, bgDark

load_dotenv()
WEBHOOK = os.getenv("FEEDBACK_WEBHOOK")
GIT_TOKEN = os.getenv("GIT_TOKEN")

models = ["Implied Volatility", "Extrapolation", "Aggregate-Extrapolation", "Logical Analysis [UNAVAILABLE]"]

humanizer = functions.Humanizer()
git = Github(auth=Auth.Token(GIT_TOKEN))

def getVersion():
    RELEASE = 1
    try:
        user = git.get_user()
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
        title=f"{round(info.getCurrentPrice(),2)} ({humanizer.sign(round(info.getPriceChange(),2))}%)"
    )
    embed.set_author(name=f"{str.upper(ticker)}")
    embed.add_field(name=f"Open: {static.get('getDayOpen'):.2f}", value=f"Close*: {static.get('getDayClose'):.2f}", inline=True)
    embed.add_field(name=f"High: {info.getDayHigh():.2f}", value=f"Low: {info.getDayLow():.2f}", inline=True)
    embed.add_field(name=f"52W H: {static.get('get52wkHigh'):.2f}", value=f"52W L: {static.get('get52wkLow'):.2f}", inline=True)

    embed.add_field(name=f"Volume: {humanizer.suffix(static.get('getVolume'))}", value=f"Avg Volume: {humanizer.suffix(static.get('getAvgVolume'))}", inline=True)
    embed.add_field(name=f"P/E: {static.get('getPERatio'):.2f}", value=f"EPS: {static.get('getEPSRatio'):.2f}", inline=True)
    embed.add_field(name=f"Beta: {static.get('getBeta'):.2f}", value=f"Mkt Cap: {humanizer.suffix(static.get('getMktCap'))}", inline=True)

    embed.add_field(name=f"Annual Yield: {static.get('getAnnualYield')}%", value=f"Monthly Yield: {static.get('getMonthlyYield')}%", inline=True)
    embed.add_field(name=f"Ex. Div.: {static.get('getExDividendDate')}", value=f"Div. Payout: {static.get('getPayDate')}")
    embed.add_field(name=f"Expected Amount: {static.get('getDividendAmount')}", value=f"Expected Change: {static.get('getDividendChange')}")
    embed.set_footer(text="* is previous day's close, with the exception of aftermarket, whereby 'Close' is the day's close")
    return embed

class Update(discord.ui.View):
    def __init__(self, ticker):
        super().__init__(timeout=None)
        self.ticker = ticker
    
    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.gray, custom_id="Refresh")
    async def refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
        new_info = functions.yFinanceWrapper(ticker=self.ticker)
        new_static = getStatic(new_info)
        
        await interaction.response.edit_message(embed=infoEmbed(info=new_info, ticker=self.ticker, static=new_static), view=self)

class Feedback(discord.ui.View):
    def __init__(self, alertPrice, alertTicker, model, fileObject):
        super().__init__(timeout=None)
        self.alertPrice = alertPrice
        self.ticker = alertTicker
        self.model = model
        self.file = fileObject

    async def feedback(self, interaction: discord.Interaction, rating):
        self.file.seek(0)
        hook = DiscordWebhook(url=WEBHOOK, content=f"Rating: **{rating}**, Version: {getVersion()}, Ticker: {self.ticker}, Model: {self.model}, Timestamp: {datetime.datetime.now()}") 
        hook.add_file(file=self.file, filename="output.png")
        await asyncio.to_thread(hook.execute)
        
        items_to_remove = [child for child in self.children if isinstance(child, discord.ui.Button) and child.custom_id in ("LikeButton", "DislikeButton")]
        for item in items_to_remove: self.remove_item(item)
        await interaction.edit_original_response(view=self)

    @discord.ui.button(label="Set Alert", style=discord.ButtonStyle.green, custom_id="AlertButton")
    async def alert(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await interaction.followup.send(f"```An alert has been set at ${self.alertPrice} for {self.ticker}.```", ephemeral=True)

    @discord.ui.button(label="Realistic", style=discord.ButtonStyle.gray, custom_id="LikeButton")
    async def likeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.feedback(interaction, "Realistic")

    @discord.ui.button(label="Unrealistic", style=discord.ButtonStyle.gray, custom_id="DislikeButton")
    async def dislikeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.feedback(interaction, "Unrealistic")

class Robot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    def lookup(self, query, header="Results for", boolean=False):
        results = yf.Lookup(query).get_all(10)
        symbols = results.index.to_list() if "exchange" not in results.columns else results["exchange"].index.to_list()
        names = results["shortName"] if "shortName" in results else None
        sanity = query.upper() in [s.upper() for s in symbols]
        if boolean: return sanity
        if results is None or results.empty or results.index.empty: return discord.Embed(color=discord.Colour.teal(),title=f"No suggestions found. Please check your spelling.")

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

    @app_commands.command(name="help", description="List all commands, and additional information")
    async def help(self, interaction: discord.Interaction):
        await interaction.response.defer()
        embed = discord.Embed(color=discord.Colour.teal(), title=f"Quantum (v{getVersion()})")
        txt = open("index\modular\help.txt","r")
        embed.description = f"""{txt.read()}"""
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="quote", description="Provide latest quote of a given ticker only")
    @app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)")
    async def quote(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()

        sanity = self.lookup(ticker,boolean=True)
        if sanity == False:
            await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
            return

        info = functions.yFinanceWrapper(ticker=ticker)
        static = getStatic(info)
        
        update = Update(ticker=ticker)
        embed = infoEmbed(info=info, ticker=ticker, static=static) if sanity else None
        await interaction.followup.send(f"Here is the current data {interaction.user.mention}:", embed=embed, view=update)

    @app_commands.command(name="chart", description="Provide latest chart and quote of a given ticker")
    @app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)", duration="Time range of data to display on the graph")
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
    async def chart(self, interaction: discord.Interaction, ticker: str, duration:str):
        charts = functions.Charts()
        await interaction.response.defer()
        
        sanity = self.lookup(ticker,boolean=True)
        if sanity == False:
            await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"))
            return

        info = functions.yFinanceWrapper(ticker=ticker)
        static = getStatic(info)

        update = Update(ticker=ticker)
        embed = infoEmbed(info=info, ticker=ticker, static=static) if sanity else None
        
        invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
        icon = interaction.guild.icon.url if interaction.guild.icon else "bot/assets/placeholderIcon.jpg"

        img = await asyncio.to_thread(charts.history, ticker, duration, interaction.guild.name, invite.url, icon)
        if img:
            file = discord.File(img, filename="output.png")
            embed.set_image(url="attachment://output.png")
            await interaction.followup.send(f"Here is today's charts {interaction.user.mention}:", file=file, embed=embed, view=update)

    @app_commands.command(name="alerts", description="Create/check/clear alerts for your given ticker")
    @app_commands.describe(ticker="Ticker to create/delete alerts for")
    async def alerts(self, interaction: discord.Interaction, ticker: typing.Optional[app_commands.Choice[str]]):
        pass

    @app_commands.command(name="predict", description="Predicts future movements of a given ticker")
    @app_commands.describe(ticker="The ticker symbol to predict (ex. AAPL)", model="Choose model algorithm")
    @app_commands.choices(model=[
        app_commands.Choice(name=models[0], value="0"),
        app_commands.Choice(name=models[1], value="1"),
        app_commands.Choice(name=models[2], value="2"),
        app_commands.Choice(name=models[3], value="3")])
    async def predict(self, interaction: discord.Interaction, ticker: str, model: typing.Optional[app_commands.Choice[str]]):
        #try:
            charts = functions.Charts()
            await interaction.response.defer()

            if self.lookup(ticker,boolean=True) == False:
                await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return

            embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} Prediction (3mo)")
            embed.set_footer(text=f"Every piece of feedback will be considered and any feedback will help improve the prediction models.")
            
            selectedModel = int(model.value) if model else 2
            warning = False

            invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
            icon = interaction.guild.icon.url if interaction.guild.icon else "bot/assets/placeholderIcon.jpg"

            img = await asyncio.to_thread(charts.project, ticker, selectedModel, interaction.guild.name, invite.url, icon)
            
            if img is None:
                warning = True
                img = await asyncio.to_thread(charts.project, ticker, 1, interaction.guild.name, invite.url, icon)
            if img:
                img_copy = io.BytesIO(img.getvalue())
                file = discord.File(img, filename="output.png")
                embed.set_image(url="attachment://output.png")
                
                feedback_view = Feedback(90, ticker, selectedModel, img_copy)
                if warning: embed.description = "Model has been changed because there were not enough datapoints to draw an accurate conclusion."
                
                await interaction.followup.send(f"Here is today's predictions ({models[int(selectedModel if not warning else 1)]} Model) {interaction.user.mention}:", file=file, embed=embed, view=feedback_view)
        #except Exception as e:
            #traceback.print_exc()
            #await interaction.followup.send("```An error occurred on our part. Please try again. If the problem persists, please contact support.```", ephemeral=True)

    @app_commands.command(name="tickers", description="Check/find the exact ticker for a given query")
    @app_commands.describe(query="The input to validate")
    async def tickers(self, interaction: discord.Interaction, query:str):
        await interaction.response.defer()
        await interaction.followup.send(embed=self.lookup(query=query))

async def setup(bot): await bot.add_cog(Robot(bot))