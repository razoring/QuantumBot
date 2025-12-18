# --- START OF FILE instance.py ---
import os
import sys # <--- REQUIRED for reloading modules
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
        try:
            await self.load_extension("cogs.robot")
            print("Extension 'cogs.robot' loaded.")
        except Exception as e:
            print(f"ERROR: {e}")
            
    async def on_ready(self):
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="/predict"))
        print(f"Logged in as {self.user} (ID: {self.user.id})")

bot = QuantumBot()

@bot.command(name="reload", hidden=True)
@commands.is_owner()
async def reload(ctx):
    status_msg = await ctx.send("Reloading...")
    
    try:
        await bot.unload_extension("cogs.robot")
        nuked = ["cogs.functions", "themes"]
        
        for module in nuked:
            if module in sys.modules:
                del sys.modules[module]
        await bot.load_extension("cogs.robot")
        
        if ctx.guild:
            bot.tree.copy_global_to(guild=ctx.guild)
            await bot.tree.sync(guild=ctx.guild)
        
        await status_msg.edit(content="Reloaded & Synced (Local Guild) [Use !sync to sync globally]")
        
    except Exception as e:
        print(f"Reload Error: {e}")
        await status_msg.edit(content=f"ERROR: ```{e}```")

@bot.command(name="sync", hidden=True)
@commands.is_owner()
async def globalsync(ctx):
    msg = await ctx.send("Syncing Globally... (This may take up to 1hr to appear)")
    await bot.tree.sync()
    await msg.edit(content="Global Sync Complete.")

if __name__ == "__main__":
    bot.run(TOKEN)