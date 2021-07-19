#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import gradio as gr

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

def compose_predictions_grid(images):
    cols = 4
    rows = len(images) // cols
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (w * cols, h * rows))
    for i, img_ in enumerate(images):
        row = i // cols
        col = i % cols
        img.paste(img_, (w * col, h * row))
    return img

def top_k_predictions_real(prompt, num_candidates=32, k=8):
    images = hallucinate(prompt, num_images=num_candidates)
    images = clip_top_k(prompt, images, k=num_preds)
    return images

def top_k_predictions(prompt, num_candidates=32, k=8):
    images = []
    for i in range(k):
        image = Image.open(f"sample_images/image_{i}.jpg")
        images.append(image)
    return images

def run_inference(prompt, num_images=32, num_preds=8):
    images = top_k_predictions(prompt, num_candidates=num_images, k=num_preds)
    predictions = compose_predictions(images)
    output_title = f"""
    <p style="font-size:22px; font-style:bold">Best predictions</p>
    <p>We asked our model to generate 32 candidates for your prompt:</p>

    <pre>

    <b>{prompt}</b>
    </pre>
    <p>We then used a pre-trained CLIP model to score them according to the
    similarity of their text and image representations.</p>

    <p>This is the result:</p>
    """
    output_description = """
    <p>Read more about the process <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA">in our report</a>.<p>
    <p style='text-align: center'>Created with <a href="https://github.com/borisdayma/dalle-mini">DALLE·mini</a></p>
    """
    return (output_title, predictions, output_description)

outputs = [
    gr.outputs.HTML(label=""),      # To be used as title
    gr.outputs.Image(label=''),
    gr.outputs.HTML(label=""),      # Additional text that appears in the screenshot
]

description = """
Welcome to our demo of DALL·E-mini. This project was created on TPU v3-8s during the 🤗 Flax / JAX Community Week.
It reproduces the essential characteristics of OpenAI's DALL·E, at a fraction of the size.

Please, write what you would like the model to generate, or select one of the examples below.
"""
gr.Interface(run_inference, 
    inputs=[gr.inputs.Textbox(label='Prompt')], #, gr.inputs.Slider(1,64,1,8, label='Candidates to generate'), gr.inputs.Slider(1,8,1,1, label='Best predictions to show')], 
    outputs=outputs, 
    title='DALL·E mini',
    description=description,
    article="<p style='text-align: center'> DALLE·mini by Boris Dayma et al. | <a href='https://github.com/borisdayma/dalle-mini'>GitHub</a></p>",
    layout='vertical',
    theme='huggingface',
    examples=[['an armchair in the shape of an avocado'], ['snowy mountains by the sea']],
    allow_flagging=False,
    live=False,
    server_port=8999
).launch(
    share=True     # Creates temporary public link if true
)
