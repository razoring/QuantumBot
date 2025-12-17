from PIL import Image, ImageFont, ImageDraw, ImageFilter
import textwrap

serverName = "A Discord Server"
serverInvite = "https://discord.gg/KeAPydnD9e"
serverIcon = "index/assets/icon.jpg"

def font(size:int):
    return ImageFont.truetype(font="index/assets/Montserrat-Bold.ttf", size=size)

main = Image.open("index/assets/main.png").convert("RGBA")
legend = Image.open("index/assets/legend.png").convert("RGBA")
chart = Image.open("index/assets/output.webp").resize((2400,1200)).convert("RGBA")
mask = Image.open("index/assets/mask.png").convert("RGBA")
blur = chart.filter(ImageFilter.BoxBlur(10))

img = Image.new(mode="RGB", size=(2500,1500), color=(10, 19, 27))
serverIcon = Image.open(serverIcon).convert("RGBA").resize((93,93))
img.paste(chart, (50,250), mask=chart)
img.paste(blur.crop(box=(24,24,2254,959)), (24,224), mask=mask)
img.paste(serverIcon, (895,76), serverIcon)
img.paste(legend, (24,224), legend)
img.paste(main, (0,0), main)
canvas = ImageDraw.Draw(im=img)
canvas.text(xy=(1003,75), text=textwrap.shorten(serverName, width=15, placeholder="..."), font=font(48), fill="white")
canvas.text(xy=(1003,135), text=textwrap.shorten(serverInvite.replace("https://",""), width=30, placeholder="..."), font=font(28), fill=(112,128,144))
img.show()