#!/usr/bin/env python
# coding: utf-8
import os

import gradio as gr
from backend import get_images_from_backend

block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")
backend_url = os.environ["BACKEND_SERVER"] + "/generate"


def infer(prompt):
    response = get_images_from_backend(prompt, backend_url)
    return response["images"]


with block:
    gr.Markdown("<h1><center>DALL·E mini</center></h1>")
    gr.Markdown(
        "DALL·E mini is an AI model that generates images from any prompt you give!"
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    margin=False,
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[3], height="auto"
        )
        text.submit(infer, inputs=text, outputs=gallery)
        btn.click(infer, inputs=text, outputs=gallery)

    gr.Markdown(
        """___
   <p style='text-align: center'>
   Created by <a href="https://twitter.com/borisdayma" target="_blank">Boris Dayma</a> et al. 2021-2022
   <br/>
   <a href="https://github.com/borisdayma/dalle-mini" target="_blank">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini-Generate-images-from-any-text-prompt--VmlldzoyMDE4NDAy" target="_blank">Project Report</a>
   </p>"""
    )


block.launch(enable_queue=False)
