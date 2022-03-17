# coding=utf-8
# Copyright 2021-2022 The Fairseq Authors and The Google Flax Team Authors And The HuggingFace Inc. team and & DALL·E Mini team. All rights reserved.
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
import os
from functools import partial
from pickle import UnpicklingError
from typing import Dict, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import unfreeze
from flax.linen import make_causal_mask
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import (
    FLAX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
)
from transformers.generation_flax_utils import FlaxSampleOutput
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
from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


class FlaxBartAttention(FlaxBartAttention):
    """
    Edits:
    - causal mask is used only in decoder and considers image_length
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
                jnp.ones((1, self.config.image_length), dtype="bool"), dtype="bool"
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
            nn.remat(FlaxBartEncoderLayer, concrete=True)
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
            nn.remat(FlaxBartDecoderLayer, concrete=True)
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
    - use image_length instead of max_position_embeddings
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
            self.config.image_length + self.offset,  # image length for BOS
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
    - init weights on CPU with `load_on_cpu`
    - restore weights on CPU with custom `from_pretrained`
    """

    config_class = DalleBartConfig

    def __init__(
        self,
        config: DalleBartConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        abstract_init: bool = False,
        load_on_cpu: bool = False,
        init_weights: bool = True,
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

        if init_weights:
            # get shape of params only
            random_params = self.init_weights(
                self.key,
                input_shape,
                abstract_init=abstract_init,
                load_on_cpu=load_on_cpu,
            )

            # save required_params as set
            self._required_params = set(flatten_dict(unfreeze(random_params)).keys())
            self.params = random_params

    def init_weights(
        self, rng=None, input_shape=(1, 1), abstract_init=False, load_on_cpu=False
    ):
        if rng is None:
            rng = self.key
        init_fn = super().init_weights
        if load_on_cpu:
            init_fn = jax.jit(init_fn, static_argnums=(1,), backend="cpu")
        if abstract_init:
            # only set shape and dtype, load parameters separately
            init_fn = partial(init_fn, input_shape=input_shape)
            params = jax.eval_shape(init_fn, rng)
        else:
            params = init_fn(rng, input_shape)
        return params

    @property
    def num_params(self):
        num_params = jax.tree_map(
            lambda param: param.size, flatten_dict(unfreeze(self.params))
        ).values()
        return sum(list(num_params))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {
            "file_type": "model",
            "framework": "flax",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Add the dtype to model_kwargs
        model_kwargs["dtype"] = dtype

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, WEIGHTS_NAME
                    )
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, FLAX_WEIGHTS_NAME
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {[FLAX_WEIGHTS_NAME, WEIGHTS_NAME]} found in directory "
                        f"{pretrained_model_name_or_path} or `from_pt` set to False"
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
                pretrained_model_name_or_path
            ):
                archive_file = pretrained_model_name_or_path
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=WEIGHTS_NAME if from_pt else FLAX_WEIGHTS_NAME,
                    revision=revision,
                )

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
                    f"  (make sure '{pretrained_model_name_or_path}' is not a path to a local directory with something else, in that case)\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(
                    f"loading weights file {archive_file} from cache at {resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # init random models
        model = cls(config, *model_args, **model_kwargs)

        with open(resolved_archive_file, "rb") as state_f:
            try:
                state = from_bytes(cls, state_f.read())
            except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                try:
                    with open(resolved_archive_file) as f:
                        if f.read().startswith("version"):
                            raise OSError(
                                "You seem to have cloned a repository without having git-lfs installed. Please install "
                                "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                                "you cloned."
                            )
                        else:
                            raise ValueError from e
                except (UnicodeDecodeError, ValueError):
                    raise EnvironmentError(
                        f"Unable to convert {archive_file} to Flax deserializable object. "
                    )

        # if model is base model only use model_prefix key
        if (
            cls.base_model_prefix not in dict(model.params)
            and cls.base_model_prefix in state
        ):
            state = state[cls.base_model_prefix]

        # if model is head model and we are loading weights from base model
        # we initialize new params dict with base_model_prefix
        if (
            cls.base_model_prefix in dict(model.params)
            and cls.base_model_prefix not in state
        ):
            state = {cls.base_model_prefix: state}

        # flatten dicts
        state = flatten_dict(state)

        random_state = flatten_dict(unfreeze(model.params))

        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        for key in state.keys():
            if key in random_state and state[key].shape != random_state[key].shape:
                if ignore_mismatched_sizes:
                    mismatched_keys.append(
                        (key, state[key].shape, random_state[key].shape)
                    )
                    state[key] = random_state[key]
                else:
                    raise ValueError(
                        f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                        f"{state[key].shape} which is incompatible with the model shape {random_state[key].shape}. "
                        "Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
                        "model."
                    )

        # add missing keys as random parameters
        for missing_key in missing_keys:
            state[missing_key] = random_state[missing_key]

        # remove unexpected keys to not be saved again
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized because the shapes did not match:\n{mismatched_warning}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )

        # set correct parameters
        model.params = unflatten_dict(state)

        return model


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
            self.config.image_vocab_size
            + 1,  # image vocab size + 1 for BOS to have same size as decoder inputs (for sharding)
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


@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]
    model_kwargs_uncond: Dict[str, jnp.ndarray]


class DalleBart(
    PretrainedFromWandbMixin, FlaxBartPreTrainedModel, FlaxBartForConditionalGeneration
):
    """
    Edits:
    - renamed from FlaxBartForConditionalGeneration
    - uses custom FlaxBartPreTrainedModel
    - uses custom FlaxBartForConditionalGenerationModule
    - no bias in decode method
    - custom prepare_inputs_for_generation using "max_length - 1" to avoid issues
      related to position embedding during model.generate()
    - custom generate method to allow super conditions
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

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jnp.DeviceArray] = None,
        decoder_attention_mask: Optional[jnp.DeviceArray] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length - 1, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length - 1), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def generate(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jnp.ndarray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        min_length: Optional[int] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        condition_scale: Optional[float] = 1.0,
        input_ids_uncond: Optional[jnp.ndarray] = None,
        attention_mask_uncond: Optional[jnp.ndarray] = None,
        **model_kwargs,
    ):
        """Edit: Allow super conditioning."""

        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id
            else self.config.decoder_start_token_id
        )
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError(
                "`decoder_start_token_id` has to be defined for encoder-decoder generation."
            )

        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs_input = dict(model_kwargs)
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    input_ids,
                    params,
                    {"attention_mask": attention_mask, **model_kwargs_input},
                )
                if condition_scale != 1.0:
                    assert (
                        input_ids_uncond is not None
                    ), "`input_ids_uncond` has to be defined for super conditioning."
                    assert (
                        do_sample is True
                    ), "`do_sample` has to be True for super conditioning."
                    assert (
                        num_beams == 1
                    ), "`num_beams` has to be 1 for super conditioning."
                    model_kwargs_uncond = (
                        self._prepare_encoder_decoder_kwargs_for_generation(
                            input_ids_uncond,
                            params,
                            {
                                "attention_mask": attention_mask_uncond,
                                **model_kwargs_input,
                            },
                        )
                    )
                else:
                    model_kwargs_uncond = None
            # prepare decoder_input_ids for generation
            input_ids = (
                jnp.ones((input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
            )

        if not do_sample and num_beams == 1:
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
            )
            return self._greedy_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif do_sample and num_beams == 1:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature
            )
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
            )
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
                condition_scale=condition_scale,
                model_kwargs_uncond=model_kwargs_uncond,
            )
        elif not do_sample and num_beams > 1:
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"][
                    "last_hidden_state"
                ] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"],
                    num_beams=num_beams,
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=num_beams
                )

            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
            )

            return self._beam_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor=None,
        logits_warper=None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
        condition_scale: float = 1.0,
        model_kwargs_uncond: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id)
        pad_token_id = jnp.array(pad_token_id)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(
            input_ids, max_length, **model_kwargs
        )
        if condition_scale != 1.0:
            model_kwargs_uncond = self.prepare_inputs_for_generation(
                input_ids, max_length, **model_kwargs_uncond
            )

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
            model_kwargs_uncond=model_kwargs_uncond,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(
                has_reached_max_length, all_sequence_finished
            )
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(
                state.running_token, params=params, **state.model_kwargs
            )

            logits = model_outputs.logits[:, -1]

            # perform super conditioning
            # Source: @RiversHaveWings - https://twitter.com/RiversHaveWings/status/1478093658716966912?s=20&t=xdm-wZ61Wf7OLnE_NJHZ1w
            if condition_scale != 1.0:
                model_outputs_uncond = model(
                    state.running_token, params=params, **state.model_kwargs_uncond
                )
                logits_uncond = model_outputs_uncond.logits[:, -1]
                logits = logits_uncond + condition_scale * (logits - logits_uncond)
            else:
                model_outputs_uncond = None

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_k, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)

            next_is_sent_finished = state.is_sent_finished | (
                next_token == eos_token_id
            )
            next_token = (
                next_token * ~next_is_sent_finished
                + pad_token_id * next_is_sent_finished
            )
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(
                state.sequences, next_token, (0, state.cur_len)
            )
            next_model_kwargs = self.update_inputs_for_generation(
                model_outputs, state.model_kwargs
            )
            next_model_kwargs_uncond = (
                self.update_inputs_for_generation(
                    model_outputs_uncond, state.model_kwargs_uncond
                )
                if condition_scale != 1.0
                else None
            )

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                model_kwargs_uncond=next_model_kwargs_uncond,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(
                sample_search_cond_fn, sample_search_body_fn, state
            )
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return FlaxSampleOutput(sequences=state.sequences)
