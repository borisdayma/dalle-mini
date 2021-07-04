# JAX implementation of VQGAN from taming-transformers https://github.com/CompVis/taming-transformers

from functools import partial
from typing import Tuple, Optional
import math

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers import PretrainedConfig


### VQGAN CONFIGURATION   ###
# https://github.com/patil-suraj/vqgan-jax/blob/main/configuration_vqgan.py
#######################

class VQGANConfig(PretrainedConfig):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        in_channels: int = 3,
        num_res_blocks: int = 2,
        resolution: int = 256,
        z_channels: int = 256,
        ch_mult: Tuple = (1, 1, 2, 2, 4),
        attn_resolutions: Tuple[int] = (16,),
        n_embed: int = 1024,
        embed_dim: int = 256,
        dropout: float = 0.0,
        double_z: bool = False,
        resamp_with_conv: bool = True,
        give_pre_end: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ch = ch
        self.out_ch = out_ch
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        self.ch_mult = list(ch_mult)
        self.attn_resolutions = list(attn_resolutions)
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.double_z = double_z
        self.resamp_with_conv = resamp_with_conv
        self.give_pre_end = give_pre_end
        self.num_resolutions = len(ch_mult)



### VQGAN MODEL ###
# https://github.com/patil-suraj/vqgan-jax/blob/main/modeling_flax_vqgan.py
#######################

class Upsample(nn.Module):
    in_channels: int
    with_conv: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample(nn.Module):
    in_channels: int
    with_conv: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states):
        if self.with_conv:
            pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
            hidden_states = jnp.pad(hidden_states, pad_width=pad)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = nn.avg_pool(hidden_states, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return hidden_states


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: Optional[int] = None
    use_conv_shortcut: bool = False
    temb_channels: int = 512
    dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv1 = nn.Conv(
            features=self.out_channels_,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        if self.temb_channels:
            self.temb_proj = nn.Dense(features=self.out_channels_, dtype=self.dtype)

        self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.dropout = nn.Dropout(rate=self.dropout_prob)
        self.conv2 = nn.Conv(
            features=self.out_channels_,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv(
                    features=self.out_channels_,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    dtype=self.dtype,
                )
            else:
                self.nin_shortcut = nn.Conv(
                    features=self.out_channels_,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="VALID",
                    dtype=self.dtype,
                )

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(nn.swish(temb))[:, :, None, None]  # TODO: check shapes

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return hidden_states + residual


class AttnBlock(nn.Module):
    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        conv = partial(
            nn.Conv, self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dtype=self.dtype
        )

        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.q, self.k, self.v = conv(), conv(), conv()
        self.proj_out = conv()

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        # compute attentions
        batch, height, width, channels = query.shape
        query = query.reshape((batch, height * width, channels))
        key = key.reshape((batch, height * width, channels))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query, key)
        attn_weights = attn_weights * (int(channels) ** -0.5)
        attn_weights = nn.softmax(attn_weights, axis=2)

        ## attend to values
        value = value.reshape((batch, height * width, channels))
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)
        hidden_states = hidden_states.reshape((batch, height, width, channels))

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class UpsamplingBlock(nn.Module):
    config: VQGANConfig
    curr_res: int
    block_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.ch * self.config.ch_mult[-1]
        else:
            block_in = self.config.ch * self.config.ch_mult[self.block_idx + 1]

        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            res_blocks.append(
                ResnetBlock(
                    block_in, block_out, temb_channels=self.temb_ch, dropout_prob=self.config.dropout, dtype=self.dtype
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in, dtype=self.dtype))

        self.block = res_blocks
        self.attn = attn_blocks

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, with_conv=self.config.resamp_with_conv, dtype=self.dtype)

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        for res_block in self.block:
            hidden_states = res_block(hidden_states, temb, deterministic=deterministic)
            for attn_block in self.attn:
                hidden_states = attn_block(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownsamplingBlock(nn.Module):
    config: VQGANConfig
    curr_res: int
    block_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        block_in = self.config.ch * in_ch_mult[self.block_idx]
        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(
                ResnetBlock(
                    in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout_prob=self.config.dropout, dtype=self.dtype
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in, dtype=self.dtype))

        self.block = res_blocks
        self.attn = attn_blocks

        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(in_channels=block_in, with_conv=self.config.resamp_with_conv, dtype=self.dtype)

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        for res_block in self.block:
            hidden_states = res_block(hidden_states, temb, deterministic=deterministic)
            for attn_block in self.attn:
                hidden_states = attn_block(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class MidBlock(nn.Module):
    in_channels: int
    temb_channels: int
    dropout: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.block_1 = ResnetBlock(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout,
            dtype=self.dtype,
        )
        self.attn_1 = AttnBlock(in_channels=self.in_channels, dtype=self.dtype)
        self.block_2 = ResnetBlock(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout,
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        hidden_states = self.block_1(hidden_states, temb, deterministic=deterministic)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states, temb, deterministic=deterministic)
        return hidden_states


class Encoder(nn.Module):
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.temb_ch = 0

        # downsampling
        self.conv_in = nn.Conv(
            self.config.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        curr_res = self.config.resolution
        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, curr_res, block_idx=i_level, dtype=self.dtype))

            if i_level != self.config.num_resolutions - 1:
                curr_res = curr_res // 2
        self.down = downsample_blocks

        # middle
        mid_channels = self.config.ch * self.config.ch_mult[-1]
        self.mid = MidBlock(mid_channels, self.temb_ch, self.config.dropout, dtype=self.dtype)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(
            2 * self.config.z_channels if self.config.double_z else self.config.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, pixel_values, deterministic: bool = True):
        # timestep embedding
        temb = None

        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states, temb, deterministic=deterministic)

        # middle
        hidden_states = self.mid(hidden_states, temb, deterministic=deterministic)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.temb_ch = 0

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.config.ch * self.config.ch_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv(
            block_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # middle
        self.mid = MidBlock(block_in, self.temb_ch, self.config.dropout, dtype=self.dtype)

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, curr_res, block_idx=i_level, dtype=self.dtype))
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = list(reversed(upsample_blocks))  # reverse to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(
            self.config.out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, deterministic: bool = True):
        # timestep embedding
        temb = None

        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states, temb, deterministic=deterministic)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states, temb, deterministic=deterministic)

        # end
        if self.config.give_pre_end:
            return hidden_states

        hidden_states = self.norm_out(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = nn.Embed(self.config.n_embed, self.config.embed_dim, dtype=self.dtype)  # TODO: init

    def __call__(self, hidden_states):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        #  flatten
        hidden_states_flattended = hidden_states.reshape((-1, self.config.embed_dim))

        # dummy op to init the weights, so we can access them below
        self.embedding(jnp.ones((1, 1), dtype="i4"))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        distance = (
            jnp.sum(hidden_states_flattended ** 2, axis=1, keepdims=True)
            + jnp.sum(emb_weights ** 2, axis=1)
            - 2 * jnp.dot(hidden_states_flattended, emb_weights.T)
        )

        # get quantized latent vectors
        min_encoding_indices = jnp.argmin(distance, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute the codebook_loss (q_loss) outside the model
        # here we return the embeddings and indices
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        z_q = self.embedding(indices)
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1)
        return z_q


class VQModule(nn.Module):
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = Encoder(self.config, dtype=self.dtype)
        self.decoder = Decoder(self.config, dtype=self.dtype)
        self.quantize = VectorQuantizer(self.config, dtype=self.dtype)
        self.quant_conv = nn.Conv(
            self.config.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        self.post_quant_conv = nn.Conv(
            self.config.z_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    def encode(self, pixel_values, deterministic: bool = True):
        hidden_states = self.encoder(pixel_values, deterministic=deterministic)
        hidden_states = self.quant_conv(hidden_states)
        quant_states, indices = self.quantize(hidden_states)
        return quant_states, indices

    def decode(self, hidden_states, deterministic: bool = True):
        hidden_states = self.post_quant_conv(hidden_states)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)
        return hidden_states

    def decode_code(self, code_b):
        hidden_states = self.quantize.get_codebook_entry(code_b)
        hidden_states = self.decode(hidden_states)
        return hidden_states

    def __call__(self, pixel_values, deterministic: bool = True):
        quant_states, indices = self.encode(pixel_values, deterministic)
        hidden_states = self.decode(quant_states, deterministic)
        return hidden_states, indices


class VQGANPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = VQGANConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: VQGANConfig,
        input_shape: Tuple = (1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, pixel_values)["params"]

    def encode(self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params}, jnp.array(pixel_values), not train, rngs=rngs, method=self.module.encode
        )

    def decode(self, hidden_states, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(hidden_states),
            not train,
            rngs=rngs,
            method=self.module.decode,
        )

    def decode_code(self, indices, params: dict = None):
        return self.module.apply(
            {"params": params or self.params}, jnp.array(indices, dtype="i4"), method=self.module.decode_code
        )

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
    ):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values),
            not train,
            rngs=rngs,
        )


class VQModel(VQGANPreTrainedModel):
    module_class = VQModule


## Utility testing ###
from PIL import Image


def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = np.clip(x, -1., 1.)
  x = (x + 1.)/2.
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def preprocess(img:Image.Image, target_image_size=256,):
    s = min((img.width, img.height))
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    # img = Image.resize(img, s, interpolation=Image.LANCZOS)
    img = img.resize(s, resample=Image.LANCZOS)
    # Center crop
    def crop_center(pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                             (img_height - crop_height) // 2,
                             (img_width + crop_width) // 2,
                             (img_height + crop_height) // 2))

    # img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = crop_center(img, target_image_size, target_image_size)
    # img = torch.unsqueeze(T.ToTensor()(img), 0)
    return np.asarray(img)[np.newaxis, :]/255.

if __name__ == "__main__":
    image = "0810.png"
    size=384
    image = Image.open(image)
    import ipdb; ipdb.set_trace()
    image = preprocess(image, size)

    model = VQModel.from_pretrained("valhalla/vqgan-imagenet-f16-1024")

    quant_states, indices = model.encode(image)
    rec = model.decode(quant_states)

    custom_to_pil(preprocess_vqgan(image[0])).save("orig.png")
    custom_to_pil(preprocess_vqgan(np.asarray(rec[0]))).save("rec.png")
