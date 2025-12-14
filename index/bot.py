import os
import typing
from dotenv import load_dotenv
import tracemalloc

import discord
from discord.ext import commands
from discord import app_commands

import functions
from themes import brand, bgDark

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(intents=intents, command_prefix="!")

models = ["Implied Volatility", "Extrapolation", "Aggregate-Extrapolation", "Logical Analysis [UNAVAILABLE]"]

"""TODO:
- Make a feedback system (using webhooks)
- Make an error-safe yfinance library of the most common types of data
- Make a stocks info command
- Stock news from yahoo finance
- Make a points system
- Implement the AI
- Implement the help system
- Telemetry data (get github version, and save options data [preferably use data with a date range])
- Compare command
- Alert command
- Release version with github
"""

projection = functions.Projection()

@bot.event
async def on_ready():
    await bot.tree.sync()

@bot.tree.command(name="help", description="Prints debug information.")
async def help(interaction: discord.Interaction):
    await interaction.response.send_message(f"Responsive Investment Calculation Heuristic (R.I.C.H.)")

@bot.tree.command(name="info", description="Provide latest quote and news of a given ticker")
@app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)")
async def info(interaction: discord.Interaction, ticker: str):
    pass

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

    embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} (90 day prediction)")
    #embed.set_footer(text=f"{interaction.user.mention}")
    if type(model) is not type(None):
        selectedModel = int(model.value)
    else:
        selectedModel = 2

    warning = False
    image_buffer = projection.create(ticker, selectedModel)
    if image_buffer is None:
        warning = True
        image_buffer = projection.create(ticker, 1)
    if image_buffer:
        file = discord.File(image_buffer, filename="output.png")
        embed.set_image(url="attachment://output.png")

        if warning == True:
            embed.description = "Model has been changed because there were not enough datapoints to draw an accurate conclusion."

        await interaction.followup.send(f"Here is today's predictions ({models[int(selectedModel if warning == False else 1)]} Model) {interaction.user.mention}:",file=file, embed=embed)
    else:
        await interaction.followup.send("```ERROR: Please check you entered the ticker symbol correct.```")

bot.run(TOKEN)