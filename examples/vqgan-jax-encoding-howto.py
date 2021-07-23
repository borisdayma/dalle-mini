#!/usr/bin/env python
# coding: utf-8

# VQGAN-JAX - Encoding HowTo

import numpy as np

# For data loading
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode

# For data saving
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import jax
from jax import pmap

from vqgan_jax.modeling_flax_vqgan import VQModel

## Params and arguments

image_list = '/sddata/dalle-mini/CC12M/10k.tsv'     # List of paths containing images to encode
output_tsv = 'output.tsv'                           # Encoded results
batch_size = 64
num_workers = 4    # TPU v3-8s have 96 cores, so feel free to increase this number when necessary

# Load model
model = VQModel.from_pretrained("flax-community/vqgan_f16_16384")

## Data Loading.

# Simple torch Dataset to load images from paths.
# You can use your own pipeline instead.
class ImageDataset(Dataset):
    def __init__(self, image_list_path: str, image_size: int, max_items=None):
        """
        :param image_list_path: Path to a file containing a list of all images. We assume absolute paths for now.
        :param image_size: Image size. Source images will be resized and center-cropped.
        :max_items: Limit dataset size for debugging
        """
        self.image_list = pd.read_csv(image_list_path, sep='\t', header=None)
        if max_items is not None: self.image_list = self.image_list[:max_items]
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_list)
    
    def _get_raw_image(self, i):
        image_path = Path(self.image_list.iloc[i][0])
        return default_loader(image_path)
    
    def resize_image(self, image):
        s = min(image.size)
        r = self.image_size / s
        s = (round(r * image.size[1]), round(r * image.size[0]))
        image = TF.resize(image, s, interpolation=InterpolationMode.LANCZOS)
        image = TF.center_crop(image, output_size = 2 * [self.image_size])
        image = np.expand_dims(np.array(image), axis=0)
        return image
    
    def __getitem__(self, i):
        image = self._get_raw_image(i)
        return self.resize_image(image)

## Encoding

# Encoding function to be parallelized with `pmap`
# Note: images have to be square
def encode(model, batch):
    _, indices = model.encode(batch)
    return indices

# Alternative: create a batch with num_tpus*batch_size and use `shard` to distribute.
def superbatch_generator(dataloader, num_tpus):
    iter_loader = iter(dataloader)
    for batch in iter_loader:
        superbatch = [batch.squeeze(1)]
        try:
            for _ in range(num_tpus-1):
                batch = next(iter_loader)
                if batch is None:
                    break
                # Skip incomplete last batch
                if batch.shape[0] == dataloader.batch_size:
                    superbatch.append(batch.squeeze(1))
        except StopIteration:
            pass
        superbatch = torch.stack(superbatch, axis=0)
        yield superbatch

def encode_dataset(dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    superbatches = superbatch_generator(dataloader, num_tpus=jax.device_count())
    
    num_tpus = jax.device_count()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    superbatches = superbatch_generator(dataloader, num_tpus=num_tpus)
    
    p_encoder = pmap(lambda batch: encode(model, batch))

    # We save each superbatch to avoid reallocation of buffers as we process them.
    # We keep the file open to prevent excessive file seeks.
    with open(output_tsv, "w") as file:
        iterations = len(dataset) // (batch_size * num_tpus)
        for n in tqdm(range(iterations)):
            superbatch = next(superbatches)
            encoded = p_encoder(superbatch.numpy())
            encoded = encoded.reshape(-1, encoded.shape[-1])

            # Extract paths from the dataset, and save paths and encodings (as string) to disk
            start_index = n * batch_size * num_tpus
            end_index = (n+1) * batch_size * num_tpus
            paths = dataset.image_list[start_index:end_index][0].values
            encoded_as_string = list(map(lambda item: np.array2string(item, separator=',', max_line_width=50000, formatter={'int':lambda x: str(x)}), encoded))
            batch_df = pd.DataFrame.from_dict({"image_file": paths, "encoding": encoded_as_string})
            batch_df.to_csv(file, sep='\t', header=(n==0), index=None)
            
dataset = ImageDataset(image_list, image_size=256)
encoded_dataset = encode_dataset(dataset, batch_size=batch_size)
