from PIL import Image, ImageDraw, ImageFont

def captioned_strip(images, caption=None, rows=1):
    increased_h = 0 if caption is None else 48
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images)*w//rows, h*rows + increased_h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i//rows*w, increased_h + (i % rows) * h))

    if caption is not None:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 40)
        draw.text((20, 3), caption, (255,255,255), font=font)
    return img
