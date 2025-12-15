import os
import typing
from dotenv import load_dotenv
from discord_webhook import DiscordWebhook
from urllib.parse import urlparse as url

import discord
from discord.ext import commands
from discord import app_commands

import functions
from themes import brand, bgDark

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK = os.getenv("FEEDBACK_WEBHOOK")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(intents=intents, command_prefix="!")

models = ["Implied Volatility", "Extrapolation", "Aggregate-Extrapolation", "Logical Analysis [UNAVAILABLE]"]

"""TODO:
- Stock news from yahoo finance
- Make a points system
- Implement the AI
- Implement the help system
- Telemetry data (get github version, and save options data [preferably use data with a date range])
- Compare command
- Alert command
- Release version with github
- BACKTEST DATA
- Feedback needs to actually work
- Caching system
"""

projection = functions.Projection()
humanizer = functions.Humanizer()

class Update(discord.ui.View):
    def __init__(self, ticker, static, info):
        super().__init__(timeout=None)
        self.ticker = ticker
        self.static = static
        self.info = info
    
    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.gray, custom_id="Refresh")
    async def refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
        #info = functions.yFinanceWrapper(ticker=self.ticker)
        await interaction.message.edit(embed=infoEmbed(info=self.info, ticker=self.ticker, static=self.static), view=self)

class Feedback(discord.ui.View):
    def __init__(self, alertPrice, alertTicker):
        super().__init__(timeout=None)
        # store for use in callbacks
        self.alertPrice = alertPrice
        self.alertTicker = alertTicker

    @discord.ui.button(label="Set Alert", style=discord.ButtonStyle.green, custom_id="AlertButton")
    async def alert(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await interaction.followup.send(f"```An alert has been set at ${self.alertPrice} for {self.alertTicker}. (ID: )```", ephemeral=True)

    @discord.ui.button(label="Realistic", style=discord.ButtonStyle.gray, custom_id="LikeButton")
    async def likeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        hook = DiscordWebhook(url=WEBHOOK, content="test")
        response = hook.execute()
        if response:
            await interaction.followup.send("```Thank you for your feedback! We will review the model soon.```", ephemeral=True)
        else:
            await interaction.followup.send("```Sorry your request could not be processed at this time.```", ephemeral=True)
        # remove both feedback buttons (Like and Dislike), keep Alert button
        for child in list(self.children):
            if isinstance(child, discord.ui.Button) and child.custom_id in ("LikeButton", "DislikeButton"):
                self.remove_item(child)
        # edit the original message to update the view
        await interaction.message.edit(view=self)

    @discord.ui.button(label="Unrealistic", style=discord.ButtonStyle.gray, custom_id="DislikeButton")
    async def dislikeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        hook = DiscordWebhook(url=WEBHOOK, content="test")
        response = hook.execute()
        if response:
            await interaction.followup.send("```Thank you for the feedback! We will try to adjust the model soon.```", ephemeral=True)
        else:
            await interaction.followup.send("```Sorry your request could not be processed at this time.```", ephemeral=True)
        # remove both feedback buttons (Like and Dislike), keep Alert button
        for child in list(self.children):
            if isinstance(child, discord.ui.Button) and child.custom_id in ("LikeButton", "DislikeButton"):
                self.remove_item(child)
        await interaction.message.edit(view=self)

@bot.event
async def on_ready():
    await bot.tree.sync()

@bot.tree.command(name="help", description="Prints debug information.")
async def help(interaction: discord.Interaction):
    await interaction.response.send_message(f"Responsive Investment Calculation Heuristic (R.I.C.H.)")

def infoEmbed(info:any, ticker:str, static:dict):
    embed = discord.Embed(color=discord.Colour.teal(), title=f"{round(info.getCurrentPrice(),2)} ({humanizer.sign(round(info.getPriceChange(),2))}%)")
    embed.set_author(name=f"{str.upper(ticker)}")
    embed.add_field(name=f"Open: {static.get('getDayOpen'):.2f}",value=f"Close*: {static.get('getDayClose'):.2f}",inline=True)
    embed.add_field(name=f"High: {info.getDayHigh():.2f}",value=f"Low: {info.getDayLow():.2f}",inline=True)
    embed.add_field(name=f"52W H: {static.get('get52wkHigh'):.2f}",value=f"52W L: {static.get('get52wkLow'):.2f}",inline=True)

    embed.add_field(name=f"Volume: {humanizer.suffix(static.get('getVolume'))}",value=f"Avg Volume: {humanizer.suffix(static.get('getAvgVolume'))}",inline=True)
    embed.add_field(name=f"P/E: {static.get('getPERatio'):.2f}",value=f"EPS: {static.get('getEPSRatio'):.2f}",inline=True)
    embed.add_field(name=f"Beta: {static.get('getBeta'):.2f}",value=f"Mkt Cap: {humanizer.suffix(static.get('getMktCap'))}",inline=True)

    embed.add_field(name=f"Annual Yield: {static.get('getAnnualYield')}%",value=f"Monthly Yield: {static.get('getMonthlyYield')}%",inline=True)
    embed.add_field(name=f"Ex. Div.: {static.get('getExDividendDate')}",value=f"Div. Payout: {static.get('getPayDate')}")
    embed.add_field(name=f"Expected Amount: {static.get('getDividendAmount')}", value=f"Expected Change: {static.get('getDividendChange')}")
    embed.set_footer(text="* is previous day's close, with the exception of aftermarket, whereby 'Close' is the day's close")
    return embed

@bot.tree.command(name="quote", description="Provide latest chart and quote of a given ticker")
@app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)", duration="Time range of data to display on the graph")
async def quote(interaction: discord.Interaction, ticker: str, duration:str):
    await interaction.response.defer()
    info = functions.yFinanceWrapper(ticker=ticker)

    static = {
        "getDayOpen":info.getDayOpen(),
        "getDayClose":info.getDayClose(),
        "get52wkHigh":info.get52wkHigh(),
        "get52wkLow":info.get52wkLow(),
        "getVolume":info.getVolume(),
        "getAvgVolume":info.getAvgVolume(),
        "getPERatio":info.getPERatio(),
        "getEPSRatio":info.getEPSRatio(),
        "getBeta":info.getBeta(),
        "getMktCap":info.getMktCap(),
        "getAnnualYield":info.getAnnualYield(),
        "getMonthlyYield":info.getMonthlyYield(),
        "getExDividendDate":info.getExDividendDate(),
        "getPayDate":info.getPayDate(),
        "getDividendAmount":info.getDividendAmount(),
        "getDividendChange":info.getDividendChange(),
    }

    update = Update(info=info,ticker=ticker,static=static)
    embed = infoEmbed(info=info,ticker=ticker,static=static)
    await interaction.followup.send(f"Here is the current charts {interaction.user.mention}", embed=embed, view=update)

@bot.tree.command(name="chart", description="Provide latest chart and quote of a given ticker")
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
async def chart(interaction: discord.Interaction, ticker: str, duration:str):
    await interaction.response.defer()
    info = functions.yFinanceWrapper(ticker=ticker)

    static = {
        "getDayOpen":info.getDayOpen(),
        "getDayClose":info.getDayClose(),
        "get52wkHigh":info.get52wkHigh(),
        "get52wkLow":info.get52wkLow(),
        "getVolume":info.getVolume(),
        "getAvgVolume":info.getAvgVolume(),
        "getPERatio":info.getPERatio(),
        "getEPSRatio":info.getEPSRatio(),
        "getBeta":info.getBeta(),
        "getMktCap":info.getMktCap(),
        "getAnnualYield":info.getAnnualYield(),
        "getMonthlyYield":info.getMonthlyYield(),
        "getExDividendDate":info.getExDividendDate(),
        "getPayDate":info.getPayDate(),
        "getDividendAmount":info.getDividendAmount(),
        "getDividendChange":info.getDividendChange(),
    }

    update = Update(info=info,ticker=ticker,static=static)
    embed = infoEmbed(info=info,ticker=ticker,static=static)
    await interaction.followup.send(f"Here is the current charts {interaction.user.mention}", embed=embed, view=update)

@bot.tree.command(name="alerts", description="Create or check alerts for your given ticker")
@app_commands.describe(action="Action to take", ticker="Ticker to create/delete alerts for", price = "Price to set alert for", identifier = "Identifier used for deletion")
@app_commands.choices(action=[
    app_commands.Choice(name="Create", value="c"),
    app_commands.Choice(name="Delete", value="d"),
    app_commands.Choice(name="List", value="l"),
    app_commands.Choice(name="Clear", value="c")
])
async def alerts(interaction: discord.Interaction, ticker: typing.Optional[app_commands.Choice[str]], action: str, price: typing.Optional[app_commands.Choice[str]], identifier: typing.Optional[app_commands.Choice[str]]):
    if type(ticker) is type(None) and type(price) is type(None) and type(identifier) is type(None):
        await interaction.response.send_message("```Nothing but us chickens. (See /help for help)```", ephemeral=True)
    await interaction.response.defer()

@bot.tree.command(name="predict", description="Predicts future movements of a given ticker")
@app_commands.describe(ticker="The ticker symbol to predict (ex. AAPL)", model="Choose model algorithm")
@app_commands.choices(model=[
    app_commands.Choice(name=models[0], value="0"),
    app_commands.Choice(name=models[1], value="1"),
    app_commands.Choice(name=models[2], value="2"),
    app_commands.Choice(name=models[3], value="3")])
async def predict(interaction: discord.Interaction, ticker: str, model: typing.Optional[app_commands.Choice[str]]):
    await interaction.response.defer()
    feedback = Feedback(90, ticker)

    embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} Prediction (3mo)")
    embed.set_footer(text=f"Every piece of feedback will be considered and any feedback will help improve the prediction models.")
    if type(model) is not type(None):
        selectedModel = int(model.value)
    else:
        selectedModel = 2

    warning = False
    img = projection.create(ticker, selectedModel)
    if img is None:
        warning = True
        img = projection.create(ticker, 1)
    if img:
        file = discord.File(img, filename="output.png")
        embed.set_image(url="attachment://output.png")
        if warning == True:
            embed.description = "Model has been changed because there were not enough datapoints to draw an accurate conclusion."
        await interaction.followup.send(f"Here is today's predictions ({models[int(selectedModel if warning == False else 1)]} Model) {interaction.user.mention}:",file=file, embed=embed, view=feedback)
    else:
        await interaction.followup.send("```ERROR: Please check you entered the ticker symbol correct.```", view=feedback)

bot.run(TOKEN)