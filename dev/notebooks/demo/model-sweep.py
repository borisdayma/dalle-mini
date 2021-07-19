#!/usr/bin/env python
# coding: utf-8

import random

import jax
import flax.linen as nn
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate

from transformers.models.bart.modeling_flax_bart import *
from transformers import BartTokenizer, FlaxBartForConditionalGeneration

import io

import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from dalle_mini.vqgan_jax.modeling_flax_vqgan import VQModel

# TODO: set those args in a config file
OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'facebook/bart-large-cnn'
WANDB_MODEL = '3iwhu4w6'

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

tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)
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

# ## CLIP Scoring
from transformers import CLIPProcessor, FlaxCLIPModel

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
    outputs = clip(**inputs)
    logits = outputs.logits_per_text
    scores = np.array(logits[0]).argsort()[-k:][::-1]
    return [images[score] for score in scores]

from PIL import ImageDraw, ImageFont

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
    strips = []
    for prompt in prompts:
        print(f"Generating candidates for: {prompt}")
        images = hallucinate(prompt, num_images=32)
        selected = clip_top_k(prompt, images, k=8)
        strip = captioned_strip(selected, prompt)
        strips.append(wandb.Image(strip))
    wandb.log({"images": strips})

## Artifact loop

import wandb
import os
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

id = wandb.util.generate_id()
print(f"Logging images to wandb run id: {id}")

run = wandb.init(id=id,
        entity='wandb',
        project="hf-flax-dalle-mini",
        job_type="predictions",
        resume="allow"
)

artifact = run.use_artifact('wandb/hf-flax-dalle-mini/model-3iwhu4w6:v0', type='bart_model')
producer_run = artifact.logged_by()
logged_artifacts = producer_run.logged_artifacts()

for artifact in logged_artifacts:
    print(f"Generating predictions with version {artifact.version}")
    artifact_dir = artifact.download()

    # create our model
    model = CustomFlaxBartForConditionalGeneration.from_pretrained(artifact_dir)
    model.config.force_bos_token_to_be_generated = False
    model.config.forced_bos_token_id = None
    model.config.forced_eos_token_id = None

    bart_params = replicate(model.params)
    vqgan_params = replicate(vqgan.params)

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
