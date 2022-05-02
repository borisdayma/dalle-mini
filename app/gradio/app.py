#!/usr/bin/env python
# coding: utf-8
import os

os.system("pip install gradio==2.9b15")

import gradio as gr
from backend import get_images_from_backend

block = gr.Blocks()
backend_url = os.environ["BACKEND_SERVER"] + "/generate"


def infer(prompt):
    response = get_images_from_backend(prompt, backend_url)
    return response["images"]


with block:
    gr.Markdown("<h1><center>DALL·E mini</center></h1>")
    gr.Markdown(
        "DALL·E mini is an AI model that generates images from any prompt you give!"
    )
    prompt = gr.inputs.Textbox(
        placeholder="An astronaut riding a horse in a photorealistic style"
    )
    text_run = gr.Button("Run")
    result = gr.Gallery()
    text_run.click(infer, inputs=prompt, outputs=result)

    gr.Markdown(
        """___
<p style='text-align: center'>
Created by Boris Dayma et al. 2021-2022
<br/>
<a href="https://github.com/borisdayma/dalle-mini" target="_blank">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA" target="_blank">Project Report</a>
</p>"""
    )


block.launch()
