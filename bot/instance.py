import os
import sys
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Establish single, persistent database connection at startup
import psycopg2 as pg
try:
    import threading
    DB_CONNECTION = pg.connect(
        dbname="QuantumBot",
        user=os.getenv("PG_USERNAME"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST", "172.17.0.1")
    )
    DB_LOCK = threading.RLock()
except Exception as e:
    print(f"Database Initialization Critical Error: {e}")
    sys.exit(1)

class QuantumBot(commands.Bot):
    def __init__(self, intents):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # Inject persistent DB objects into modules before loading extensions
        import functions
        functions.DB_CONNECTION = DB_CONNECTION
        functions.DB_LOCK = DB_LOCK
        await self.load_extension("cogs.robot")
        print("Extension 'cogs.robot' loaded.")
            
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        if self.intents.members:
            totalUsers = sum(g.member_count for g in self.guilds)
            print(f"Total Users: {totalUsers}")
        else:
            print("Total Users: Unknown (Members intent disabled)")

        from cogs.robot import ServerAccess, VERIFY_CHANNEL_ID
        channel = self.get_channel(VERIFY_CHANNEL_ID)
        if channel:
            try:
                await channel.purge()
                embed = discord.Embed(color=discord.Colour.teal(), title="Getting Started")
                embed.description = "## Server Access\nClick the button below to register your account and access the private channels."
                await channel.send(embed=embed, view=ServerAccess())
            except Exception as e:
                print(f"Failed to send startup message: {e}")

        if not self._updateStatus.is_running():
            self._updateStatus.start()

    @tasks.loop(minutes=30)
    async def _updateStatus(self):
        if self.intents.members:
            totalUsers = sum(g.member_count for g in self.guilds)
            name = f"{totalUsers} traders | /help"
        else:
            name = "/help"
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=name))

def run_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.invites = True
    intents.members = True

    try:
        qBot = QuantumBot(intents)
        
        @qBot.command(name="reload", hidden=True)
        @commands.is_owner()
        async def reloadExtensions(ctx):
            statusMsg = await ctx.send("Reloading...")
            await qBot.unload_extension("cogs.robot")
            _targetModules = ["functions", "themes", "cogs.robot"]
            for moduleName in _targetModules:
                if moduleName in sys.modules:
                    del sys.modules[moduleName]
            import functions
            import themes
            functions.DB_CONNECTION = DB_CONNECTION
            functions.DB_LOCK = DB_LOCK
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

        print("Attempting to connect with privileged intents...")
        qBot.run(TOKEN)
    except discord.errors.PrivilegedIntentsRequired:
        print("Privileged intents denied. Falling back to basic configuration.")
        intents.message_content = False
        intents.members = False
        fallbackBot = QuantumBot(intents)
        fallbackBot.run(TOKEN)

if __name__ == "__main__": 
    run_bot()
