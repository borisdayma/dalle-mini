# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" DalleBart model configuration """
import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DalleBartConfig(PretrainedConfig):
    model_type = "dallebart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    def __init__(
        self,
        normalize_text=False,
        encoder_vocab_size=50264,
        image_vocab_size=16384,  # encoded image token space
        image_length=256,  # number of encoded tokens
        max_text_length=64,  # max number of text tokens
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        gradient_checkpointing=False,
        use_cache=True,
        is_encoder_decoder=True,
        forced_eos_token_id=None,
        tie_word_embeddings=False,  # different modalities and sizes
        **kwargs,
    ):
        self.normalize_text = normalize_text
        self.encoder_vocab_size = encoder_vocab_size
        self.image_vocab_size = image_vocab_size
        self.image_length = image_length
        self.max_text_length = max_text_length
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )

        # remove inferred keys to prevent errors when loading config (passed as kwargs)
        for k in [
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "decoder_start_token_id",
            "min_length",
            "max_length",
        ]:
            kwargs.pop(k, None)

        super().__init__(
            pad_token_id=image_vocab_size
            + 1,  # needed to avoid errors during generation (converted to jnp.array)
            bos_token_id=image_vocab_size + 1,  # set to unreachable values
            eos_token_id=image_vocab_size + 1,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=image_vocab_size,  # BOS appended to vocab
            forced_eos_token_id=forced_eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            min_length=image_length + 1,
            max_length=image_length + 1,
            **kwargs,
        )

        # ensure backward compatibility for BART CNN models
        if self.forced_bos_token_id is None and kwargs.get(
            "force_bos_token_to_be_generated", False
        ):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions."
                "The config can simply be saved and uploaded again to be fixed."
            )
