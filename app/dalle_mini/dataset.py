"""
An image-caption dataset dataloader.
Luke Melas-Kyriazi, 2021
"""
import warnings
from typing import Optional, Callable
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import ImageFile
from PIL.Image import DecompressionBombWarning
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DecompressionBombWarning)


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class for (image, texts) tasks. Note that this dataset 
    returns the raw text rather than tokens. This is done on purpose, because
    it's easy to tokenize a batch of text after loading it from this dataset.
    """

    def __init__(self, *, images_root: str, captions_path: str, text_transform: Optional[Callable] = None, 
                 image_transform: Optional[Callable] = None, image_transform_type: str = 'torchvision',
                 include_captions: bool = True):
        """
        :param images_root: folder where images are stored
        :param captions_path: path to csv that maps image filenames to captions
        :param image_transform: image transform pipeline
        :param text_transform: image transform pipeline
        :param image_transform_type: image transform type, either `torchvision` or `albumentations`
        :param include_captions: Returns a dictionary with `image`, `text` if `true`; otherwise returns just the images.
        """

        # Base path for images
        self.images_root = Path(images_root)

        # Load captions as DataFrame
        self.captions = pd.read_csv(captions_path, delimiter='\t', header=0)
        self.captions['image_file'] = self.captions['image_file'].astype(str)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.image_transform_type = image_transform_type.lower()
        assert self.image_transform_type in ['torchvision', 'albumentations']

        # Total number of datapoints
        self.size = len(self.captions)

        # Return image+captions or just images
        self.include_captions = include_captions

    def verify_that_all_images_exist(self):
        for image_file in self.captions['image_file']:
            p = self.images_root / image_file
            if not p.is_file():
                print(f'file does not exist: {p}')

    def _get_raw_image(self, i):
        image_file = self.captions.iloc[i]['image_file']
        image_path = self.images_root / image_file
        image = default_loader(image_path)
        return image

    def _get_raw_text(self, i):
        return self.captions.iloc[i]['caption']

    def __getitem__(self, i):
        image = self._get_raw_image(i)
        caption = self._get_raw_text(i)
        if self.image_transform is not None:
            if self.image_transform_type == 'torchvision':
                image = self.image_transform(image)
            elif self.image_transform_type == 'albumentations':
                image = self.image_transform(image=np.array(image))['image']
            else:
                raise NotImplementedError(f"{self.image_transform_type=}")
        return {'image': image, 'text': caption} if self.include_captions else image

    def __len__(self):
        return self.size


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from transformers import AutoTokenizer
    
    # Paths
    images_root = './images'
    captions_path = './images-list-clean.tsv'

    # Create transforms
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    def tokenize(text):
        return tokenizer(text, max_length=32, truncation=True, return_tensors='pt', padding='max_length')
    image_transform = A.Compose([
        A.Resize(256, 256), A.CenterCrop(256, 256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()])
    
    # Create dataset
    dataset = CaptionDataset(
        images_root=images_root,
        captions_path=captions_path,
        image_transform=image_transform,
        text_transform=tokenize,
        image_transform_type='albumentations')

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    print({k: (v.shape if isinstance(v, torch.Tensor) else v) for k, v in batch.items()})

    # # (Optional) Check that all the images exist
    # dataset = CaptionDataset(images_root=images_root, captions_path=captions_path)
    # dataset.verify_that_all_images_exist()
    # print('Done')
