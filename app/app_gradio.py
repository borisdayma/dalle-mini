#!/usr/bin/env python
# coding: utf-8

# Uncomment to run on cpu
#import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import random

import jax
import flax.linen as nn
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate

from transformers.models.bart.modeling_flax_bart import *
from transformers import BartTokenizer, FlaxBartForConditionalGeneration


import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


from dalle_mini.vqgan_jax.modeling_flax_vqgan import VQModel

import gradio as gr


# TODO: set those args in a config file
OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'flax-community/dalle-mini'

class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # we keep shared to easily load pre-trained weights
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        # a separate embedding is used for the decoder
        self.decoder_embed = nn.Embed(
            OUTPUT_VOCAB_SIZE,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

        # the decoder has a different config
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = OUTPUT_LENGTH
        decoder_config.vocab_size = OUTPUT_VOCAB_SIZE
        self.decoder = FlaxBartDecoder(decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed)

class CustomFlaxBartForConditionalGenerationModule(FlaxBartForConditionalGenerationModule):
    def setup(self):
        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            OUTPUT_VOCAB_SIZE,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, OUTPUT_VOCAB_SIZE))

class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule

# create our model
# FIXME: Save tokenizer to hub so we can load from there
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = CustomFlaxBartForConditionalGeneration.from_pretrained(BASE_MODEL)
model.config.force_bos_token_to_be_generated = False
model.config.forced_bos_token_id = None
model.config.forced_eos_token_id = None

vqgan = VQModel.from_pretrained("flax-community/vqgan_f16_16384")

def custom_to_pil(x):
    x = np.clip(x, 0., 1.)
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def generate(input, rng, params):
    return model.generate(
        **input,
        max_length=257,
        num_beams=1,
        do_sample=True,
        prng_key=rng,
        eos_token_id=50000,
        pad_token_id=50000,
        params=params,
    )

def get_images(indices, params):
    return vqgan.decode_code(indices, params=params)

def plot_images(images):
    fig = plt.figure(figsize=(40, 20))
    columns = 4
    rows = 2
    plt.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    
def stack_reconstructions(images):
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images)*w, h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i*w,0))
    return img

p_generate = jax.pmap(generate, "batch")
p_get_images = jax.pmap(get_images, "batch")

bart_params = replicate(model.params)
vqgan_params = replicate(vqgan.params)

# ## CLIP Scoring
from transformers import CLIPProcessor, FlaxCLIPModel

clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("Initialize FlaxCLIPModel")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Initialize CLIPProcessor")

def hallucinate(prompt, num_images=64):
    prompt = [prompt] * jax.device_count()
    inputs = tokenizer(prompt, return_tensors='jax', padding="max_length", truncation=True, max_length=128).data
    inputs = shard(inputs)

    all_images = []
    for i in range(num_images // jax.device_count()):
        key = random.randint(0, 1e7)
        rng = jax.random.PRNGKey(key)
        rngs = jax.random.split(rng, jax.local_device_count())
        indices = p_generate(inputs, rngs, bart_params).sequences
        indices = indices[:, :, 1:]

        images = p_get_images(indices, vqgan_params)
        images = np.squeeze(np.asarray(images), 1)
        for image in images:
            all_images.append(custom_to_pil(image))
    return all_images

def clip_top_k(prompt, images, k=8):
    inputs = processor(text=prompt, images=images, return_tensors="np", padding=True)
    outputs = clip(**inputs)
    logits = outputs.logits_per_text
    scores = np.array(logits[0]).argsort()[-k:][::-1]
    return [images[score] for score in scores]

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

def top_k_predictions(prompt, num_candidates=32, k=8):
    images = hallucinate(prompt, num_images=num_candidates)
    images = clip_top_k(prompt, images, k=k)
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
    <p>We then used a pre-trained <a href="https://huggingface.co/openai/clip-vit-base-patch32">CLIP model</a> to score them according to the
    similarity of the text and the image representations.</p>

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
    # server_port=8999
).launch()
