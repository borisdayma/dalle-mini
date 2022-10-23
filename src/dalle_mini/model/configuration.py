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

from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


class DalleBartConfig(PretrainedFromWandbMixin, PretrainedConfig):
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
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        gradient_checkpointing=True,
        use_scan=None,
        use_cache=True,
        is_encoder_decoder=True,
        forced_eos_token_id=None,
        tie_word_embeddings=False,  # different modalities and sizes
        do_sample=True,
        # transformer variants
        use_bias=False,  # use bias in attention and dense layers (except for lm_head)
        ln_type="layernorm",  # layer normalization type, "rmsnorm", "layernorm"
        ln_positions="normformer",  # layer normalization positions, "normformer", "swinv2", "cogview", "postln", "preln", "deepnet" (same as postln), "subln"
        use_head_scale=False,  # used in NormFormer
        use_cosine_attention=False,  # used in Swin v2
        tau_init=0.05,  # used only in cosine attention (Swin v2)
        use_absolute_position_embeddings=True,  # default
        use_swin_position_embeddings=False,  # used in Swin v1/v2
        use_deepnet_scaling=False,  # used in Deepnet
        use_subln_init=False,
        use_glu=True,  # "GLU Variants Improve Transformer"
        use_alibi=False,  # Not implemented yet - from "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
        sinkhorn_iters=1,  # used in SinkFormers
        use_final_ln_encoder=True,  # final layer normalization in encoder
        use_final_ln_decoder=True,  # final layer normalization in decoder
        # parameters that should not be necessary but could affect results
        force_ln_scale=False,  # force scale in layernorm even when followed by dense layers
        **kwargs,
    ):
        # text normalizer
        self.normalize_text = normalize_text

        # transformer variants
        self.use_bias = use_bias
        assert ln_type in [
            "rmsnorm",
            "layernorm",
        ], "ln_type must be 'rmsnorm' or 'layernorm'"
        self.ln_type = ln_type
        if ln_positions == "deepnet":
            ln_positions = "postln"
        assert ln_positions in [
            "normformer",
            "swinv2",
            "cogview",
            "postln",
            "preln",
            "subln",
        ], "ln_positions must be 'normformer', 'swinv2', 'cogview', 'postln', 'preln', 'subln'"
        self.use_head_scale = use_head_scale
        assert use_alibi is False, "use_alibi is not supported yet"
        self.ln_positions = ln_positions
        self.use_cosine_attention = use_cosine_attention
        self.tau_init = tau_init
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_swin_position_embeddings = use_swin_position_embeddings
        self.use_deepnet_scaling = use_deepnet_scaling
        self.use_subln_init = use_subln_init
        self.use_glu = use_glu
        self.use_alibi = use_alibi
        self.sinkhorn_iters = sinkhorn_iters
        if ln_positions == "postln":
            assert (
                use_final_ln_encoder
            ), "use_final_ln_encoder must be True when ln_positions is 'postln'"
            assert (
                use_final_ln_decoder
            ), "use_final_ln_decoder must be True when ln_positions is 'postln'"
        self.use_final_ln_encoder = use_final_ln_encoder
        self.use_final_ln_decoder = use_final_ln_decoder
        self.force_ln_scale = force_ln_scale

        # common parameters
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
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        # all layers are the same in most configurations
        self.use_scan = use_scan if use_scan is not None else ln_positions != "swinv2"
        assert not (
            self.use_scan and ln_positions == "swinv2"
        ), "scan cannot be used with 'swinv2'"
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )

        # special token id's are appended to vocab if not provided
        decoder_start_token_id = kwargs.pop("decoder_start_token_id", image_vocab_size)
        bos_token_id = kwargs.pop("bos_token_id", image_vocab_size)
        pad_token_id = kwargs.pop("pad_token_id", image_vocab_size)
        eos_token_id = kwargs.pop("eos_token_id", image_vocab_size)

        # we generate to image_length + 1 (for bos) by default
        min_length = kwargs.pop("min_length", image_length + 1)
        max_length = kwargs.pop("max_length", image_length + 1)

        super().__init__(
            # args required in parent class
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            forced_eos_token_id=forced_eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            min_length=min_length,
            max_length=max_length,
            do_sample=do_sample,
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
