""" DalleBart tokenizer """
from transformers import BartTokenizer
from transformers.utils import logging

from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


class DalleBartTokenizer(PretrainedFromWandbMixin, BartTokenizer):
    pass
