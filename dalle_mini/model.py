import jax
import flax.linen as nn

from transformers.models.bart.modeling_flax_bart import (
    FlaxBartModule,
    FlaxBartForConditionalGenerationModule,
    FlaxBartForConditionalGeneration,
    FlaxBartEncoder,
    FlaxBartDecoder,
)

from transformers import BartConfig


class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # we keep shared to easily load pre-trained weights
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # a separate embedding is used for the decoder
        self.decoder_embed = nn.Embed(
            self.config.image_vocab_size + 1,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.encoder = FlaxBartEncoder(
            self.config, dtype=self.dtype, embed_tokens=self.shared
        )

        # the decoder has a different config
        # TODO: should not be needed once we have custom config/module
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = (
            self.config.image_length + 1  # image tokens + BOS
        )
        decoder_config.vocab_size = self.config.image_vocab_size + 1
        self.decoder = FlaxBartDecoder(
            decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed
        )


class CustomFlaxBartForConditionalGenerationModule(
    FlaxBartForConditionalGenerationModule
):
    def setup(self):
        # set default config
        self.config.normalize_text = getattr(self.config, "normalize_text", False)
        self.config.image_length = getattr(self.config, "image_length", 256)
        self.config.image_vocab_size = getattr(self.config, "image_vocab_size", 16384)

        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.image_vocab_size + 1,  # encoded image token space + 1 for bos
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.final_logits_bias = self.param(
            "final_logits_bias", self.bias_init, (1, self.config.image_vocab_size + 1)
        )


class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule
