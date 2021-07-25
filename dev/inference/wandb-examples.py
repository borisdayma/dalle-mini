#!/usr/bin/env python
# coding: utf-8

import random

import jax
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate

from transformers.models.bart.modeling_flax_bart import *
from transformers import BartTokenizer, FlaxBartForConditionalGeneration

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from dalle_mini.model import CustomFlaxBartForConditionalGeneration
from vqgan_jax.modeling_flax_vqgan import VQModel

# ## CLIP Scoring
from transformers import CLIPProcessor, FlaxCLIPModel

import wandb
import os

from dalle_mini.helpers import captioned_strip


os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

# TODO: used for legacy support
BASE_MODEL = 'facebook/bart-large-cnn'

# set id to None so our latest images don't get overwritten
id = None
run = wandb.init(id=id,
        entity='wandb',
        project="hf-flax-dalle-mini",
        job_type="predictions",
        resume="allow"
)
artifact = run.use_artifact('wandb/hf-flax-dalle-mini/model-4oh3u7ca:latest', type='bart_model')
artifact_dir = artifact.download()

# create our model
model = CustomFlaxBartForConditionalGeneration.from_pretrained(artifact_dir)

# TODO: legacy support (earlier models)
tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)
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

clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
    # FIXME: image should be resized and normalized prior to being processed by CLIP
    outputs = clip(**inputs)
    logits = outputs.logits_per_text
    scores = np.array(logits[0]).argsort()[-k:][::-1]
    return [images[score] for score in scores]

def log_to_wandb(prompts):
    strips = []
    for prompt in prompts:
        print(f"Generating candidates for: {prompt}")
        images = hallucinate(prompt, num_images=32)
        selected = clip_top_k(prompt, images, k=8)
        strip = captioned_strip(selected, prompt)
        strips.append(wandb.Image(strip))
    wandb.log({"images": strips})

prompts = prompts = [
    "white snow covered mountain under blue sky during daytime",
    "aerial view of beach during daytime",
    "aerial view of beach at night",
    "an armchair in the shape of an avocado",
    "young woman riding her bike trough a forest",
    "rice fields by the mediterranean coast",
    "white houses on the hill of a greek coastline",
    "illustration of a shark with a baby shark",
]

log_to_wandb(prompts)
