
import jax
import flax.linen as nn

from transformers.models.bart.modeling_flax_bart import (
    FlaxBartModule,
    FlaxBartForConditionalGenerationModule,
    FlaxBartForConditionalGeneration,
    FlaxBartEncoder,
    FlaxBartDecoder
)

from transformers import BartConfig


# Model hyperparameters, for convenience
OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'facebook/bart-large-cnn'  # we currently have issues with bart-large


class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # check config is valid, otherwise set default values
        self.config.vocab_size_output = getattr(self.config, 'vocab_size_output', OUTPUT_VOCAB_SIZE)
        self.config.max_position_embeddings_decoder = getattr(self.config, 'max_position_embeddings_decoder', OUTPUT_LENGTH)

        # we keep shared to easily load pre-trained weights
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        # a separate embedding is used for the decoder
        self.decoder_embed = nn.Embed(
            self.config.vocab_size_output,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

        # the decoder has a different config
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = self.config.max_position_embeddings_decoder
        decoder_config.vocab_size = self.config.vocab_size_output
        self.decoder = FlaxBartDecoder(decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed)

class CustomFlaxBartForConditionalGenerationModule(FlaxBartForConditionalGenerationModule):
    def setup(self):
        # check config is valid, otherwise set default values
        self.config.vocab_size_output = getattr(self.config, 'vocab_size_output', OUTPUT_VOCAB_SIZE)

        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size_output,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.config.vocab_size_output))

class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule
