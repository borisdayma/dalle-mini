# coding=utf-8
# Copyright 2021 The Fairseq Authors and The Google Flax Team Authors And The HuggingFace Inc. team and the DalleBart team. All rights reserved.
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
""" DalleBart model. """

import math
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze
from flax.linen import make_causal_mask
from flax.traverse_util import flatten_dict
from jax.random import PRNGKey
from transformers.modeling_flax_outputs import (
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
)
from transformers.modeling_flax_utils import ACT2FN
from transformers.models.bart.modeling_flax_bart import (
    FlaxBartAttention,
    FlaxBartDecoder,
    FlaxBartDecoderLayer,
    FlaxBartDecoderLayerCollection,
    FlaxBartEncoder,
    FlaxBartEncoderLayer,
    FlaxBartEncoderLayerCollection,
    FlaxBartForConditionalGeneration,
    FlaxBartForConditionalGenerationModule,
    FlaxBartModule,
    FlaxBartPreTrainedModel,
)
from transformers.utils import logging

from .configuration import DalleBartConfig

logger = logging.get_logger(__name__)


class FlaxBartAttention(FlaxBartAttention):
    """
    Edits:
    - causal mask is used only in decoder and considers image_length + 1 (for BOS)
    """

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        self.dropout_layer = nn.Dropout(rate=self.dropout)

        if self.causal:
            # used only in decoder
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.image_length + 1), dtype="bool"), dtype="bool"
            )


class FlaxBartEncoderLayer(FlaxBartEncoderLayer):
    """
    Edits:
    - no bias
    - use custom FlaxBartAttention
    """

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            bias=False,
            dtype=self.dtype,
        )
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.fc2 = nn.Dense(
            self.embed_dim,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)


class FlaxBartEncoderLayerCollection(FlaxBartEncoderLayerCollection):
    """
    Edits:
    - use custom FlaxBartEncoderLayer
    - allow Gradient Checkpointing (nn.remat)
    """

    def setup(self):
        layer_module = (
            nn.remat(FlaxBartEncoderLayer)
            if self.config.gradient_checkpointing
            else FlaxBartEncoderLayer
        )
        self.layers = [
            layer_module(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        self.layerdrop = self.config.encoder_layerdrop


class FlaxBartDecoderLayer(FlaxBartDecoderLayer):
    """
    Edits:
    - no bias
    - uses custom FlaxBartAttention
    """

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            bias=False,
            dtype=self.dtype,
        )
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.encoder_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            bias=False,
            dtype=self.dtype,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.fc2 = nn.Dense(
            self.embed_dim,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)


class FlaxBartDecoderLayerCollection(FlaxBartDecoderLayerCollection):
    """
    Edits:
    - use custom FlaxBartDecoderLayer
    - allow Gradient Checkpointing (nn.remat)
    """

    def setup(self):
        layer_module = (
            nn.remat(FlaxBartDecoderLayer)
            if self.config.gradient_checkpointing
            else FlaxBartDecoderLayer
        )
        self.layers = [
            layer_module(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        self.layerdrop = self.config.decoder_layerdrop


class FlaxBartEncoder(FlaxBartEncoder):
    """
    Edits:
    - offset set to 0 (no padding token)
    - use max_text_length instead of max_position_embeddings
    - use custom FlaxBartEncoderLayerCollection
    - embed_tokens cannot be None (issue at compile time)
    """

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 0
        self.embed_positions = nn.Embed(
            self.config.max_text_length + self.offset,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.layers = FlaxBartEncoderLayerCollection(self.config, self.dtype)
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)


class FlaxBartDecoder(FlaxBartDecoder):
    """
    Edits:
    - offset set to 0 (no padding token)
    - use image_length + 1 (for BOS) instead of max_position_embeddings
    - use custom FlaxBartDecoderLayerCollection
    - embed_tokens cannot be None (issue at compile time)
    """

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.embed_scale = (
            math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0
        )

        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 0
        self.embed_positions = nn.Embed(
            self.config.image_length + 1 + self.offset,  # image length + 1 for BOS
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.layers = FlaxBartDecoderLayerCollection(self.config, self.dtype)
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)


class FlaxBartModule(FlaxBartModule):
    """
    Edits
    - use custom FlaxBartEncoder & FlaxBartDecoder
    - use separate embeddings for Encoder & Decoder
    """

    def setup(self):
        encoder_embed_tokens = nn.Embed(
            self.config.encoder_vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        decoder_embed_tokens = nn.Embed(
            self.config.image_vocab_size + 1,  # image vocab size + 1 for BOS
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.encoder = FlaxBartEncoder(
            self.config, dtype=self.dtype, embed_tokens=encoder_embed_tokens
        )
        self.decoder = FlaxBartDecoder(
            self.config, dtype=self.dtype, embed_tokens=decoder_embed_tokens
        )


class FlaxBartPreTrainedModel(FlaxBartPreTrainedModel):
    """
    Edits:
    - added num_params property
    - config_class replaced to DalleBartConfig
    - __init__ accepts abstract_init which does uses parameter shape to initialize the model
    """

    config_class = DalleBartConfig

    def __init__(
        self,
        config: DalleBartConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        abstract_init: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)

        # adapted from HuggingFace FlaxPreTrainedModel
        if config is None:
            raise ValueError("config cannot be None")

        if module is None:
            raise ValueError("module cannot be None")

        # Those are private to be exposed as typed property on derived classes.
        self._config = config
        self._module = module

        # Those are public as their type is generic to every derived classes.
        self.key = PRNGKey(seed)
        self.dtype = dtype

        # randomly initialized parameters
        if abstract_init:
            # init the model weights only abstractly, eval_shape will return a pytree
            # with the structure as weights but without any actual values, this will just contain
            # the shape information. Weights need to be loaded later.
            init_fn = partial(self.init_weights, input_shape=input_shape)
            random_params = jax.eval_shape(init_fn, self.key)
        else:
            random_params = self.init_weights(self.key, input_shape)

        # save required_params as set
        self._required_params = set(flatten_dict(unfreeze(random_params)).keys())
        self.params = random_params

    @property
    def num_params(self):
        num_params = jax.tree_map(
            lambda param: param.size, flatten_dict(unfreeze(self.params))
        ).values()
        return sum(list(num_params))


class FlaxBartForConditionalGenerationModule(FlaxBartForConditionalGenerationModule):
    """
    Edits:
    - no bias
    - lm_head set to image_vocab_size + 1 (for BOS)
    - uses custom FlaxBartModule
    """

    def setup(self):
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.image_vocab_size + 1,  # image vocab size + 1 for BOS
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_embedding.T}}, hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class DalleBart(FlaxBartPreTrainedModel, FlaxBartForConditionalGeneration):
    """
    Edits:
    - renamed from FlaxBartForConditionalGeneration
    - uses custom FlaxBartPreTrainedModel
    - uses custom FlaxBartForConditionalGenerationModule
    - no bias in decode method
    """

    module_class = FlaxBartForConditionalGenerationModule

    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `decoder_position_ids` when passing `past_key_values`."
                )

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxBartAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if self.config.tie_word_embeddings:
                shared_embedding = module.model.variables["params"]["shared"][
                    "embedding"
                ]
                lm_logits = module.lm_head.apply(
                    {"params": {"kernel": shared_embedding.T}}, hidden_states
                )
            else:
                lm_logits = module.lm_head(hidden_states)

            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs
