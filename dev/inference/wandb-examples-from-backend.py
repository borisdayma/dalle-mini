#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageDraw, ImageFont
import wandb
import os

from dalle_mini.backend import ServiceError, get_images_from_backend
from dalle_mini.helpers import captioned_strip

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

def log_to_wandb(prompts):
    try:
        backend_url = os.environ["BACKEND_SERVER"]
        for _ in range(1):        
            for prompt in prompts:
                print(f"Getting selections for: {prompt}")
                # make a separate run per prompt
                with wandb.init(
                    entity='wandb',
                    project='hf-flax-dalle-mini',
                    job_type='predictions',# tags=['openai'],
                    config={'prompt': prompt}
                ):
                    imgs = []
                    selected = get_images_from_backend(prompt, backend_url)
                    strip = captioned_strip(selected, prompt)
                    imgs.append(wandb.Image(strip))
                    wandb.log({"images": imgs})
    except ServiceError as error:
        print(f"Service unavailable, status: {error.status_code}")
    except KeyError:
        print("Error: BACKEND_SERVER unset")

prompts = [
    # "white snow covered mountain under blue sky during daytime",
    # "aerial view of beach during daytime",
    # "aerial view of beach at night",
    # "a farmhouse surrounded by beautiful flowers",
    # "an armchair in the shape of an avocado",
    # "young woman riding her bike trough a forest",
    # "a unicorn is passing by a rainbow in a field of flowers",
    # "illustration of a baby shark swimming around corals",
    # "painting of an oniric forest glade surrounded by tall trees",
    # "sunset over green mountains",
    # "a forest glade surrounded by tall trees in a sunny Spring morning",
    # "fishing village under the moonlight in a serene sunset",
    # "cartoon of a carrot with big eyes",
    # "still life in the style of Kandinsky",
    # "still life in the style of Picasso",
    # "a graphite sketch of a gothic cathedral",
    # "a graphite sketch of Elon Musk",
    # "a watercolor pond with green leaves and yellow flowers",
    # "a logo of a cute avocado armchair singing karaoke on stage in front of a crowd of strawberry shaped lamps",
    # "happy celebration in a small village in Africa",
    # "a logo of an armchair in the shape of an avocado"
    # "Pele and Maradona in a hypothetical match",
    # "Mohammed Ali and Mike Tyson in a hypothetical match",
    # "a storefront that has the word 'openai' written on it",
    # "a pentagonal green clock",
    # "a collection of glasses is sitting on a table",
    # "a small red block sitting on a large green block",
    # "an extreme close-up view of a capybara sitting in a field",
    # "a cross-section view of a walnut",
    # "a professional high-quality emoji of a lovestruck cup of boba",
    # "a photo of san francisco's golden gate bridge",
    # "an illustration of a baby daikon radish in a tutu walking a dog",
    # "a picture of the Eiffel tower on the Moon",
    # "a colorful stairway to heaven",
    "this is a detailed high-resolution scan of a human brain"
]

for _ in range(1):
    log_to_wandb(prompts)