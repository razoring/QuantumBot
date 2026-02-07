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
GIT = os.getenv("GIT_TOKEN")

# map of discord user id -> asyncio.Future used to await registration results
REGISTRATIONS: dict[int, asyncio.Future] = {}
models = ["Implied Volatility", "Extrapolation", "Aggregate-Extrapolation", "Logical Analysis [UNAVAILABLE]"]

humanizer = functions.Humanizer()
git = Github(auth=Auth.Token(GIT))

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

class Robot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

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
    
    async def authenticated(self, interaction: discord.Interaction, bypass=False):
        if self._registered(interaction.user.id) and bypass==False:
            return True

        embed = discord.Embed(color=discord.Colour.teal(),title="401: Regristration Required")
        embed.description = "You must agree to the EULA before continuing. Please take a few minutes to read the terms of service and privacy policy.\n\nThis process involves agreeing to our Terms of Service and Privacy Policy. You can review these documents using the buttons below."
        embed.add_field(name="What happens next?", value="Click 'Continue' to proceed with registration. You'll be asked about marketing communications and to confirm you've read our legal documents.", inline=False)
        try:
            if interaction.response.is_done(): await interaction.followup.send(embed=embed, view=RegisterPrompt(), ephemeral=True)
            else: await interaction.response.send_message(embed=embed, view=RegisterPrompt(), ephemeral=True)
        except Exception:
            try: await interaction.followup.send(embed=embed, view=RegisterPrompt(), ephemeral=True)
            except Exception: pass

        # create a future and wait for the modal handler to set its result
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        REGISTRATIONS[interaction.user.id] = fut

        try:
            result = await asyncio.wait_for(fut, timeout=300)
            return bool(result)
        except asyncio.TimeoutError:
            REGISTRATIONS.pop(interaction.user.id, None)
            timeout_embed = discord.Embed(color=discord.Colour.teal(), title="408: Registration Timeout")
            timeout_embed.description = "Registration timed out. Please try again."
            await interaction.followup.send(embed=timeout_embed, ephemeral=True)
            return False

    @app_commands.command(name="help", description="List all commands, and additional information")
    async def help(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            embed = discord.Embed(color=discord.Colour.teal(), title=f"Quantum (v{getVersion()})")
            with open("bot/modular/help.txt", "r") as txt:
                embed.description = txt.read()
            await interaction.followup.send(embed=embed)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, An error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="quote", description="Provide latest quote of a given ticker only")
    @app_commands.describe(ticker="The ticker symbol to return (ex. AAPL)")
    async def quote(self, interaction: discord.Interaction, ticker: str):
        try:
            await interaction.response.defer()
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            sanity = self.lookup(ticker,boolean=True)
            if sanity == False:
                await interaction.followup.send(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return

            info = functions.yFinanceWrapper(ticker=ticker)
            static = getStatic(info)
            
            update = Update(ticker=ticker)
            embed = infoEmbed(info=info, ticker=ticker, static=static) if sanity else None
            await interaction.followup.send(f"Here is the current data {interaction.user.mention}:", embed=embed, view=update)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, An error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

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
        try:
            await interaction.response.defer()
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            sanity = self.lookup(ticker,boolean=True)
            if sanity == False:
                await interaction.response.send_message(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return
            
            charts = functions.Charts()

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
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, An error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

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
        try:
            await interaction.response.defer(ephemeral=True)
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            if self.lookup(ticker,boolean=True) == False:
                await interaction.response.send_message(embed=self.lookup(query=ticker, header="Did you mean these instead of"), ephemeral=True)
                return
            
            charts = functions.Charts()

            embed = discord.Embed(color=discord.Colour.teal(), title=f"{str.upper(ticker)} Prediction (3mo)")
            embed.set_footer(text=f"Every piece of feedback will be considered and any feedback will help improve the prediction models.")
            
            selectedModel = int(model.value) if model else 2
            warning = False

            invite = await interaction.channel.create_invite(max_age=0, max_uses=0, unique=False, reason="For the advertising graphic (Quantum Bot)")
            icon = interaction.guild.icon.url if interaction.guild.icon else "bot/assets/placeholderIcon.jpg"

            loading = discord.Embed(color=discord.Colour.teal(), title="Generating Prediction...")
            loading.description = "Starting Thread..."
            status = await interaction.followup.send(embed=loading)

            loop = asyncio.get_running_loop()

            async def edit_status(text: str):
                try:
                    e = discord.Embed(color=discord.Colour.teal(), title="Generating Prediction...")
                    e.description = text
                    await status.edit(embed=e)
                except Exception: pass

            # thread-safe callback to be passed into the projection function
            def progress_cb(text: str):
                try: loop.call_soon_threadsafe(asyncio.create_task, edit_status(text))
                except Exception: pass

            img = await asyncio.to_thread(charts.project, ticker, selectedModel, interaction.guild.name, invite.url, icon, progress_cb)

            if img is None:
                warning = True
                progress_cb("Using Extrapolation Model...")
                img = await asyncio.to_thread(charts.project, ticker, 1, interaction.guild.name, invite.url, icon, progress_cb)

            if img:
                progress_cb("Finalizing/Cleaning...")
                img_copy = io.BytesIO(img.getvalue())
                file = discord.File(img_copy, filename="output.png")
                embed.set_image(url="attachment://output.png")

                feedback_view = Feedback(90, ticker, selectedModel, img_copy)
                if warning: embed.description = "WARNING: Model has been changed because there were not enough datapoints to draw an accurate conclusion."

                await interaction.followup.send(f"Here is today's predictions ({models[int(selectedModel if not warning else 1)]} Model) {interaction.user.mention}:", file=file, embed=embed, view=feedback_view)
                await status.delete()
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, An error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="tickers", description="Check/find the exact ticker for a given query (stock, index, etf, general search)")
    @app_commands.describe(query="The input to validate")
    async def tickers(self, interaction: discord.Interaction, query:str):
        try:
            await interaction.response.defer()
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            await interaction.followup.send(embed=self.lookup(query=query))
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, An error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="me", description="Display account information (hidden from others)")
    async def me(self, interaction: discord.Interaction):
        try:
            interaction.response.defer()
            if await self.authenticated(interaction=interaction, bypass=False) == False: return
            user = functions.User(discordID=interaction.user.id)
            embed = discord.Embed(color=discord.Colour.teal(), title=f"Account Information")
            embed.description = f"Discord ID: {interaction.user.id}\nUsername: {interaction.user.name}\nRegistered: Yes"
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            traceback.print_exc()
            embed = discord.Embed(color=discord.Colour.teal(), title="500: Unknown Server Error")
            embed.description = "Sorry, An error occurred on our part. Please try again. \n\nIf the problem persists, please contact support."
            await interaction.followup.send(embed=embed, ephemeral=True)

class RegisterPrompt(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="Terms of Service", url="https://github.com/razoring/QuantumDiscordBot/blob/main/TermsService"))
        self.add_item(discord.ui.Button(label="Privacy Policy", url="https://github.com/razoring/QuantumDiscordBot/blob/main/PrivacyPolicy"))

    @discord.ui.button(label="Continue", style=discord.ButtonStyle.green, custom_id="Register")
    async def register(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(RegisterModal())

class RegisterModal(discord.ui.Modal, title="Register"):
    """discord_account = discord.ui.TextInput(
        label="Discord Account",
        placeholder="wumpus",
        style=discord.TextStyle.short,
        required=True,
        custom_id="d0c1e00ef8f64ca498e6247e5ba867cb",
    )"""

    def __init__(self):
        super().__init__()

    marketing = discord.ui.Label( # use preferences to determine
        text="Marketing Communications",
        description="Confirm marketing communications (new features, exclusive promotions, etc) via Email, SMS, or DMs.",
        component=discord.ui.Select(
            custom_id="684e0b62c1c441bba9c17a4f4aa5753a",
            placeholder="Choose (You can opt-out anytime)",
            options=[
                discord.SelectOption(
                    label="AGREE",
                    value=True,
                    description="I AGREE to receive marketing communications regarding new features, promotions, and more.",
                ),
                discord.SelectOption(
                    label="DISAGREE",
                    value=False,
                    description="I DISAGREE to receive marketing communications regarding new features, promotions, and more."
                ),
            ]
        ),
    )

    legal = discord.ui.Label(
        text="EULA",
        description="Confirm you read and understand the legal disclaimers, terms, conditions, and privacy policy.",
        component=discord.ui.Select(
            custom_id="b7dd2eee99394c7fabd31ce98119d58f",
            placeholder="Choose",
            options=[
                discord.SelectOption(
                    label="AGREE",
                    value=True,
                    description="I have read, understand, and AGREE to the legal disclaimers, terms, conditions, and privacy policy."
                ),
                discord.SelectOption(
                    label="DISAGREE",
                    value=False,
                    description="I have read, and DISAGREE to the terms. Thereby I confirm that I cannot use bot's features."
                ),
            ]
        ),
    )

    """email = discord.ui.TextInput(
        label="Email",
        placeholder="email@xyz.com",
        style=discord.TextStyle.short,
        required=False,
        custom_id="919aebdb748a4220966efb7a1b8d2596",
        max_length=100,
    )

    phone = discord.ui.TextInput(
        label="Phone",
        placeholder="1-800-555-5555 (Standard rates may apply)",
        style=discord.TextStyle.short,
        required=False,
        min_length=10,
        max_length=16,
        custom_id="05daa38f551444d188c5fba6d800c658",
    )"""
    
    async def on_submit(self, interaction:discord.Interaction):
        assert isinstance(self.marketing.component, discord.ui.Select)
        assert isinstance(self.legal.component, discord.ui.Select)

        user = functions.User(interaction.user.id)
        marketing = self.marketing.component.values[0]
        legal = self.legal.component.values[0]
        # notify any waiter that registration completed (or failed)
        fut = REGISTRATIONS.pop(interaction.user.id, None)
        if legal:
            success = False
            if user.createAccount(marketing=marketing):
                success = True
                embed = discord.Embed(color=discord.Colour.teal(), title="Registration Complete")
                embed.description = "Thanks for registering! You can now use all bot features."
                # set result before replying so callers can proceed without waiting on this IO
                if fut and not fut.done():
                    fut.set_result(True)
                await interaction.response.send_message(embed=embed, ephemeral=True)
            else:
                if fut and not fut.done():
                    fut.set_result(False)
                embed = discord.Embed(color=discord.Colour.teal(), title="500: Registration Failed")
                embed.description = "An error occurred while creating your account. Please try again later."
                await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            if fut and not fut.done():
                fut.set_result(False)
            embed = discord.Embed(color=discord.Colour.teal(), title="406: Registration Failed")
            embed.description = "You must agree to the terms of service to use this bot. \n\nIf you think this is a mistake, please contact support."
            await interaction.response.send_message(embed=embed, ephemeral=True)

class Update(discord.ui.View):
    def __init__(self, ticker):
        super().__init__(timeout=300)
        self.ticker = ticker
    
    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.gray, custom_id="Refresh")
    async def refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
        new_info = functions.yFinanceWrapper(ticker=self.ticker)
        new_static = getStatic(new_info)
        
        await interaction.response.edit_message(embed=infoEmbed(info=new_info, ticker=self.ticker, static=new_static), view=self)

class Feedback(discord.ui.View):
    def __init__(self, alertPrice, alertTicker, model, fileObject):
        super().__init__(timeout=300)
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
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(f"```An alert has been set at ${self.alertPrice} for {self.ticker}.```", ephemeral=True)

    @discord.ui.button(label="Realistic", style=discord.ButtonStyle.gray, custom_id="LikeButton")
    async def likeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.feedback(interaction, "Realistic")

    @discord.ui.button(label="Unrealistic", style=discord.ButtonStyle.gray, custom_id="DislikeButton")
    async def dislikeButton(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.feedback(interaction, "Unrealistic")

# mount bot - DO NOT TOUCH
async def setup(bot): await bot.add_cog(Robot(bot))