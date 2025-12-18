# --- START OF FILE instance.py ---
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.invites = True

class QuantumBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # 1. Load the extension and catch errors
        try:
            await self.load_extension("cogs.robot")
            print("Extension 'cogs.robot' loaded successfully.")
        except Exception as e:
            print(f"Failed to load extension 'cogs.robot': {e}")

        # 2. Global Sync (Note: This takes up to 1 hour to update on Discord)
        # We use the !sync command below for instant updates during dev
        # await self.tree.sync() 

    async def on_ready(self):
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="/predict"))
        print(f"Logged in as {self.user} (ID: {self.user.id})")

bot = QuantumBot()

# --- INSTANT SYNC COMMAND ---
# Run "!sync" in your discord server to force the commands to appear immediately
@bot.command(name="sync")
@commands.is_owner()
async def sync(ctx):
    try:
        synced = await bot.tree.sync()
        await ctx.send(f"Synced {len(synced)} commands globally.")
    except Exception as e:
        await ctx.send(f"```ERROR: {e}```")

@bot.command(name="reload", hidden=True)
@commands.is_owner()
async def reload(ctx):
    print("Reload Initiating...")
    try:
        await bot.reload_extension("cogs.robot")
        # Re-sync after reload to apply changes
        await bot.tree.sync() 
        await ctx.send("'''Module Reloaded & Synced'''")
    except Exception as e:
        await ctx.send(f"'''ERROR: {e}'''")

if __name__ == "__main__":
    bot.run(TOKEN)