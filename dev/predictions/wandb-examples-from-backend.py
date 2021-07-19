#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageDraw, ImageFont
import wandb
import os

from dalle_mini.backend import ServiceError, get_images_from_backend

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

# set id to None so our latest images don't get overwritten
id = None
run = wandb.init(id=id,
        entity='wandb',
        project="hf-flax-dalle-mini",
        job_type="predictions",
        resume="allow"
)

def captioned_strip(images, caption):
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images)*w, h + 48))
    for i, img_ in enumerate(images):
        img.paste(img_, (i*w, 48))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 40)
    draw.text((20, 3), caption, (255,255,255), font=font)
    return img

def log_to_wandb(prompts):
    try:
        backend_url = os.environ["BACKEND_SERVER"]

        strips = []
        for prompt in prompts:
            print(f"Getting selections for: {prompt}")
            selected = get_images_from_backend(prompt, backend_url)
            strip = captioned_strip(selected, prompt)
            strips.append(wandb.Image(strip))
        wandb.log({"images": strips})
    except ServiceError as error:
        print(f"Service unavailable, status: {error.status_code}")
    except KeyError:
        print("Error: BACKEND_SERVER unset")

prompts = [
    "white snow covered mountain under blue sky during daytime",
    "aerial view of beach during daytime",
    "aerial view of beach at night",
    "an armchair in the shape of an avocado",
    "a logo of an avocado armchair playing music",
    "young woman riding her bike trough a forest",
    "rice fields by the mediterranean coast",
    "white houses on the hill of a greek coastline",
    "illustration of a shark with a baby shark",
    "painting of an oniric forest glade surrounded by tall trees",
]

log_to_wandb(prompts)
