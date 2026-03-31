import os
import sys
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

_intents = discord.Intents.default()
_intents.message_content = True
_intents.guilds = True
_intents.invites = True
_intents.members = True

class QuantumBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=_intents)

    async def setup_hook(self):
        await self.load_extension("cogs.robot")
        print("Extension 'cogs.robot' loaded.")
            
    async def on_ready(self):
        totalUsers = sum(g.member_count for g in self.guilds)
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print(f"Total Users: {totalUsers}")

        from cogs.robot import ServerAccess, VERIFY_CHANNEL_ID
        channel = self.get_channel(VERIFY_CHANNEL_ID)
        if channel:
            try:
                await channel.purge()
                embed = discord.Embed(color=discord.Colour.teal(), title="Server Access")
                embed.description = "## Getting Started\nClick the button below to register your account and access the private channels."
                await channel.send(embed=embed, view=ServerAccess())
            except Exception as e:
                print(f"Failed to send startup message: {e}")

        if not self._updateStatus.is_running():
            self._updateStatus.start()

    @tasks.loop(minutes=30)
    async def _updateStatus(self):
        totalUsers = sum(g.member_count for g in self.guilds)
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=f"{totalUsers} traders | /help"))

qBot = QuantumBot()

@qBot.command(name="reload", hidden=True)
@commands.is_owner()
async def reloadExtensions(ctx):
    statusMsg = await ctx.send("Reloading...")
    
    await qBot.unload_extension("cogs.robot")
    _targetModules = ["functions", "themes", "cogs.robot"]
    
    for moduleName in _targetModules:
        if moduleName in sys.modules:
            del sys.modules[moduleName]
    await qBot.load_extension("cogs.robot")
    
    if ctx.guild:
        qBot.tree.copy_global_to(guild=ctx.guild)
        await qBot.tree.sync(guild=ctx.guild)
    
    await statusMsg.edit(content="Reload complete and synced to local guild (Use !sync to sync globally)")

@qBot.command(name="sync", hidden=True)
@commands.is_owner()
async def syncGlobalTree(ctx):
    statusMsg = await ctx.send("Syncing Globally... (This may take up to 1hr to appear)")
    await qBot.tree.sync()
    await statusMsg.edit(content="Global Sync Complete")

if __name__ == "__main__": qBot.run(TOKEN)
