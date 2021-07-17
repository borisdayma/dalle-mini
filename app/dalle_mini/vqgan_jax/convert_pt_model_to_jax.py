import re

import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

import torch

from modeling_flax_vqgan import VQModel
from configuration_vqgan import VQGANConfig


regex = r"\w+[.]\d+"


def rename_key(key):
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key


# Adapted from https://github.com/huggingface/transformers/blob/ff5cdc086be1e0c3e2bbad8e3469b34cffb55a85/src/transformers/modeling_flax_pytorch_utils.py#L61
def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    random_flax_state_dict = flatten_dict(flax_model.params)
    flax_state_dict = {}

    remove_base_model_prefix = (flax_model.base_model_prefix not in flax_model.params) and (
        flax_model.base_model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )
    add_base_model_prefix = (flax_model.base_model_prefix in flax_model.params) and (
        flax_model.base_model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )

    # Need to change some parameters name to match Flax names so that we don't have to fork any layer
    for pt_key, pt_tensor in pt_state_dict.items():
        pt_tuple_key = tuple(pt_key.split("."))

        has_base_model_prefix = pt_tuple_key[0] == flax_model.base_model_prefix
        require_base_model_prefix = (flax_model.base_model_prefix,) + pt_tuple_key in random_flax_state_dict

        if remove_base_model_prefix and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]
        elif add_base_model_prefix and require_base_model_prefix:
            pt_tuple_key = (flax_model.base_model_prefix,) + pt_tuple_key

        # Correctly rename weight parameters
        if (
            "norm" in pt_key
            and (pt_tuple_key[-1] == "bias")
            and (pt_tuple_key[:-1] + ("bias",) in random_flax_state_dict)
        ):
            pt_tensor = pt_tensor[None, None, None, :]
        elif (
            "norm" in pt_key
            and (pt_tuple_key[-1] == "bias")
            and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
        ):
            pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
            pt_tensor = pt_tensor[None, None, None, :]
        elif pt_tuple_key[-1] in ["weight", "gamma"] and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict:
            pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
            pt_tensor = pt_tensor[None, None, None, :]
        if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict:
            pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
        elif pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4 and pt_tuple_key not in random_flax_state_dict:
            # conv layer
            pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
            pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        elif pt_tuple_key[-1] == "weight" and pt_tuple_key not in random_flax_state_dict:
            # linear layer
            pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
            pt_tensor = pt_tensor.T
        elif pt_tuple_key[-1] == "gamma":
            pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
        elif pt_tuple_key[-1] == "beta":
            pt_tuple_key = pt_tuple_key[:-1] + ("bias",)

        if pt_tuple_key in random_flax_state_dict:
            if pt_tensor.shape != random_flax_state_dict[pt_tuple_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[pt_tuple_key].shape}, but is {pt_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[pt_tuple_key] = jnp.asarray(pt_tensor)

    return unflatten_dict(flax_state_dict)


def convert_model(config_path, pt_state_dict_path, save_path):
    config = VQGANConfig.from_pretrained(config_path)
    model = VQModel(config)

    state_dict = torch.load(pt_state_dict_path, map_location="cpu")["state_dict"]
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("loss"):
            state_dict.pop(key)
            continue
        renamed_key = rename_key(key)
        state_dict[renamed_key] = state_dict.pop(key)

    state = convert_pytorch_state_dict_to_flax(state_dict, model)
    model.params = unflatten_dict(state)
    model.save_pretrained(save_path)
