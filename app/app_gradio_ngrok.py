#!/usr/bin/env python
# coding: utf-8

import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

import gradio as gr

# If we use streamlit, this would be exported as a streamlit secret
import os
backend_url = os.environ["BACKEND_SERVER"]

def compose_predictions(images, caption=None):
    increased_h = 0 if caption is None else 48
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images)*w, h + increased_h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i*w, increased_h))

    if caption is not None:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 40)
        draw.text((20, 3), caption, (255,255,255), font=font)
    return img

class ServiceError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code

def get_images_from_ngrok(prompt):
    r = requests.post(
        backend_url, 
        json={"prompt": prompt}
    )
    if r.status_code == 200:
        images = r.json()["images"]
        images = [Image.open(BytesIO(base64.b64decode(img))) for img in images]
        return images
    else:
        raise ServiceError(r.status_code)
        
def run_inference(prompt):
    try:
        images = get_images_from_ngrok(prompt)
        predictions = compose_predictions(images)
        output_title = f"""
        <p style="font-size:22px; font-style:bold">Best predictions</p>
        <p>We asked our model to generate 128 candidates for your prompt:</p>

        <pre>

        <b>{prompt}</b>
        </pre>
        <p>We then used a pre-trained <a href="https://huggingface.co/openai/clip-vit-base-patch32">CLIP model</a> to score them according to the
        similarity of the text and the image representations.</p>

        <p>This is the result:</p>
        """
        
        output_description = """
        <p>Read our <a style="color:blue;" href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA">full report</a> for more details on how this works.<p>
        <p style='text-align: center'>Created with <a style="color:blue;" href="https://github.com/borisdayma/dalle-mini">DALL·E mini</a></p>
        """

    except ServiceError:
        output_title = f"""
        Sorry, there was an error retrieving the images. Please, try again later or <a href="mailto:pcuenca-dalle@guenever.net">contact us here</a>.
        """
        predictions = None
        output_description = ""
        
    return (output_title, predictions, output_description)
    
outputs = [
    gr.outputs.HTML(label=""),      # To be used as title
    gr.outputs.Image(label=''),
    gr.outputs.HTML(label=""),      # Additional text that appears in the screenshot
]

description = """
Welcome to DALL·E-mini, a text-to-image generation model.
"""
gr.Interface(run_inference, 
    inputs=[gr.inputs.Textbox(label='Prompt')],
    outputs=outputs, 
    title='DALL·E mini',
    description=description,
    article="<p style='text-align: center'> DALLE·mini by Boris Dayma et al. | <a href='https://github.com/borisdayma/dalle-mini'>GitHub</a></p>",
    layout='vertical',
    theme='huggingface',
    examples=[['an armchair in the shape of an avocado'], ['snowy mountains by the sea']],
    allow_flagging=False,
    live=False,
    server_name="0.0.0.0",      # Bind to all interfaces (I think)
    # server_port=8999
).launch()
