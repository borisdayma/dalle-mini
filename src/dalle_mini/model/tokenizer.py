""" DalleBart tokenizer """
from transformers import BartTokenizerFast

from .utils import PretrainedFromWandbMixin


class DalleBartTokenizer(PretrainedFromWandbMixin, BartTokenizerFast):
    pass
