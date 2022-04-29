#!/usr/bin/env python
# coding: utf-8
import os

os.system("pip install gradio==2.9b12")
from datetime import datetime

import gradio as gr
from backend import ServiceError, get_images_from_backend
from PIL import Image

block = gr.Blocks()
backend_url = os.environ["BACKEND_SERVER"] + "/generate"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def infer(prompt):
    response = get_images_from_backend(prompt, backend_url)
    selected = response["images"]
    version = response["version"]

    images_list = []

    for i, img in enumerate(selected):
        images_list.append(img)

    grid = image_grid(images_list, rows=3, cols=3)
    print(grid)
    return grid


with block:
    gr.Markdown("<h1><center>DALL·E mini</center></h1>")
    gr.Markdown(
        "DALL·E mini is an AI model that generates images from any prompt you give!"
    )
    prompt = gr.inputs.Textbox(
        lines=3, placeholder="An astronaut riding a horse in a photorealistic style"
    )
    result = gr.outputs.Image(type="pil")
    text_run = gr.Button("Run")
    text_run.click(infer, inputs=prompt, outputs=result)

    gr.Markdown(
        """___
<p style='text-align: center'>
Created by Boris Dayma et al. 2021-2022
<br/>
<a href="https://github.com/borisdayma/dalle-mini" target="_blank">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA" target="_blank">Project Report</a>
</p>"""
    )


block.launch(enable_queue=True)
