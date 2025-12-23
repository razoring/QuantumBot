import io
import os
import typing
import datetime
from urllib.parse import urlparse as url

import discord
from discord import app_commands
from discord.ext import commands
from discord_webhook import DiscordWebhook

from github import Auth, Github
from dotenv import load_dotenv

import functions
from themes import brand, bgDark

load_dotenv()
WEBHOOK = os.getenv("FEEDBACK_WEBHOOK")
GIT_TOKEN = os.getenv("GIT_TOKEN")

models = ["Implied Volatility", "Extrapolation", "Aggregate-Extrapolation", "Logical Analysis [UNAVAILABLE]"]

charts = functions.Charts()
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

def get_static_data(info):
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
        new_static = get_static_data(new_info)
        
        await interaction.response.edit_message(
            embed=infoEmbed(info=new_info, ticker=self.ticker, static=new_static), 
            view=self
        )

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
        hook.execute()
        
        items_to_remove = [child for child in self.children if isinstance(child, discord.ui.Button) and child.custom_id in ("LikeButton", "DislikeButton")]
        for item in items_to_remove:
            self.remove_item(item)
            
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

    @app_commands.command(name="help", description="List all commands, and additional information")
    async def help(self, interaction: discord.Interaction):
        await interaction.response.defer()
        embed = discord.Embed(color=discord.Colour.teal(), title=f"Quantum (v{getVersion()})")
        txt = open("index/help.txt","r")
        embed.description = f"""{txt.read()}"""
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="quote", description="Provide latest quote of a given ticker only")
    @app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)")
    async def quote(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        info = functions.yFinanceWrapper(ticker=ticker)
        static = get_static_data(info)
        
        update_view = Update(ticker=ticker)
        embed = infoEmbed(info=info, ticker=ticker, static=static)
        await interaction.followup.send(f"Here is the current data {interaction.user.mention}:", embed=embed, view=update_view)

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
        await interaction.response.defer()
        info = functions.yFinanceWrapper(ticker=ticker)
        static = get_static_data(info)

        update_view = Update(ticker=ticker)
        embed = infoEmbed(info=info, ticker=ticker, static=static)
        
        invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
        icon = interaction.guild.icon.url if interaction.guild.icon else "index/assets/placeholderIcon.jpg"

        img = charts.history(ticker, duration, interaction.guild.name, invite.url, icon)
        if img:
            file = discord.File(img, filename="output.png")
            embed.set_image(url="attachment://output.png")
            await interaction.followup.send(f"Here is today's charts {interaction.user.mention}:", file=file, embed=embed, view=update_view)
        else:
            await interaction.followup.send("```ERROR: Please check you entered the ticker symbol correct.```", view=update_view)

    @app_commands.command(name="alerts", description="Create or check alerts for your given ticker")
    @app_commands.describe(action="Action to take", ticker="Ticker to create/delete alerts for", price="Price to set alert for", identifier="Identifier used for deletion")
    @app_commands.choices(action=[
        app_commands.Choice(name="Create", value="c"),
        app_commands.Choice(name="Delete", value="d"),
        app_commands.Choice(name="List", value="l"),
        app_commands.Choice(name="Clear", value="c")
    ])
    async def alerts(self, interaction: discord.Interaction, ticker: typing.Optional[app_commands.Choice[str]], action: str, price: typing.Optional[app_commands.Choice[str]], identifier: typing.Optional[app_commands.Choice[str]]):
        if type(ticker) is type(None) and type(price) is type(None) and type(identifier) is type(None):
            await interaction.response.send_message("```Nothing but us chickens. (See /help for help)```", ephemeral=True)
            return

    @app_commands.command(name="predict", description="Predicts future movements of a given ticker")
    @app_commands.describe(ticker="The ticker symbol to predict (ex. AAPL)", model="Choose model algorithm")
    @app_commands.choices(model=[
        app_commands.Choice(name=models[0], value="0"),
        app_commands.Choice(name=models[1], value="1"),
        app_commands.Choice(name=models[2], value="2"),
        app_commands.Choice(name=models[3], value="3")])
    async def predict(self, interaction: discord.Interaction, ticker: str, model: typing.Optional[app_commands.Choice[str]]):
        await interaction.response.defer()

        embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} Prediction (3mo)")
        embed.set_footer(text=f"Every piece of feedback will be considered and any feedback will help improve the prediction models.")
        
        selectedModel = int(model.value) if model else 2
        warning = False

        invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
        icon = interaction.guild.icon.url if interaction.guild.icon else "index/assets/placeholderIcon.jpg"

        img = charts.project(ticker, selectedModel, interaction.guild.name, invite.url, icon)
        
        if img is None:
            warning = True
            img = charts.project(ticker, 1, interaction.guild.name, invite.url, icon)
        
        if img:
            img_copy = io.BytesIO(img.getvalue())
            
            file = discord.File(img, filename="output.png")
            embed.set_image(url="attachment://output.png")
            
            feedback_view = Feedback(90, ticker, selectedModel, img_copy)
            
            if warning:
                embed.description = "Model has been changed because there were not enough datapoints to draw an accurate conclusion."
            
            await interaction.followup.send(
                f"Here is today's predictions ({models[int(selectedModel if not warning else 1)]} Model) {interaction.user.mention}:",
                file=file, embed=embed, view=feedback_view
            )
        else:
            await interaction.followup.send("```ERROR: Please check you entered the ticker symbol correct.```")
    
    @app_commands.command(name="predictiontest", description="Predicts future movements of a given ticker")
    @app_commands.describe(ticker="The ticker symbol to predict (ex. AAPL)", weights="Prediction weights for testing")
    @commands.is_owner()
    async def predictTest(self, interaction: discord.Interaction, ticker: str, weights:str):
        await interaction.response.defer()

        embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} Prediction (3mo)")
        embed.set_footer(text=f"Every piece of feedback will be considered and any feedback will help improve the prediction models.")

        img = charts.projectTest(ticker, weights)
        if img:
            file = discord.File(img, filename="output.png")
            embed.set_image(url="attachment://output.png")
            
            await interaction.followup.send(
                f"{weights}:",
                file=file, embed=embed)
        else:
            await interaction.followup.send("```ERROR: Please check you entered the ticker symbol correct.```")

async def setup(bot):
    await bot.add_cog(Robot(bot))