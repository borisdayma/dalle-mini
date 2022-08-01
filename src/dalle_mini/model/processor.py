""" DalleBart processor """

from typing import List

import jax.numpy as jnp

from .configuration import DalleBartConfig
from .text import TextNormalizer
from .tokenizer import DalleBartTokenizer
from .utils import PretrainedFromWandbMixin


class DalleBartProcessorBase:
    def __init__(
        self, tokenizer: DalleBartTokenizer, normalize_text: bool, max_text_length: int
    ):
        self.tokenizer = tokenizer
        self.normalize_text = normalize_text
        self.max_text_length = max_text_length
        if normalize_text:
            self.text_processor = TextNormalizer()
        # create unconditional tokens
        uncond = self.tokenizer(
            "",
            return_tensors="jax",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).data
        self.input_ids_uncond = uncond["input_ids"]
        self.attention_mask_uncond = uncond["attention_mask"]

    def __call__(self, text: List[str] = None):
        # check that text is not a string
        assert not isinstance(text, str), "text must be a list of strings"

        if self.normalize_text:
            text = [self.text_processor(t) for t in text]
        res = self.tokenizer(
            text,
            return_tensors="jax",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).data
        # tokens used only with super conditioning
        n = len(text)
        res["input_ids_uncond"] = jnp.repeat(self.input_ids_uncond, n, axis=0)
        res["attention_mask_uncond"] = jnp.repeat(self.attention_mask_uncond, n, axis=0)
        return res

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = DalleBartTokenizer.from_pretrained(*args, **kwargs)
        config = DalleBartConfig.from_pretrained(*args, **kwargs)
        return cls(tokenizer, config.normalize_text, config.max_text_length)


class DalleBartProcessor(PretrainedFromWandbMixin, DalleBartProcessorBase):
    pass
