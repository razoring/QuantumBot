from discord.ext import commands

class Robot(commands.Cog):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

async def setup(bot):
    await bot.add_cog(Robot(bot=bot))