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
        self.active_users = set()

    async def global_interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.command is None: return True
        if interaction.user.id in self.active_users:
            embed = discord.Embed(color=discord.Colour.teal(), title="429: Too Many Requests")
            embed.description = "You already have a command processing. Please wait for it to finish."
            try:
                if not interaction.response.is_done(): await interaction.response.send_message(embed=embed, ephemeral=True)
                else: await interaction.followup.send(embed=embed, ephemeral=True)
            except Exception: pass
            return False
        self.active_users.add(interaction.user.id)
        return True

    async def global_prefix_check(self, ctx):
        if ctx.author.id in self.active_users:
            embed = discord.Embed(color=discord.Colour.teal(), title="429: Too Many Requests")
            embed.description = "You already have a command processing. Please wait for it to finish."
            await ctx.send(embed=embed)
            return False
        self.active_users.add(ctx.author.id)
        return True

    async def on_app_command_completion(self, interaction: discord.Interaction, command):
        self.active_users.discard(interaction.user.id)

    async def on_command_completion(self, ctx):
        self.active_users.discard(ctx.author.id)

    async def on_command_error(self, ctx, error):
        self.active_users.discard(ctx.author.id)
        if hasattr(commands.Bot, "on_command_error"):
            await super().on_command_error(ctx, error)

    async def setup_hook(self):
        # Inject persistent DB objects into modules before loading extensions
        import functions
        functions.DB_CONNECTION = DB_CONNECTION
        functions.DB_LOCK = DB_LOCK

        self.tree.interaction_check = self.global_interaction_check
        self.add_check(self.global_prefix_check)

        original_on_error = self.tree.on_error
        async def on_tree_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
            self.active_users.discard(interaction.user.id)
            if original_on_error: await original_on_error(interaction, error)
        self.tree.on_error = on_tree_error

        await self.load_extension("cogs.robot")
        print("Extension 'cogs.robot' loaded.")
            
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        if self.intents.members:
            totalUsers = sum(g.member_count for g in self.guilds)
            print(f"Total Users: {totalUsers:,}")
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
            name = f"{totalUsers:,} traders | /help"
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
