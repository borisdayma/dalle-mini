import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py
# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def _get_partition_rules():
    return [
        # embeddings
        ((r"embed_positions", "embedding"), P("mp", None)),
        ((r"embed_tokens", "embedding"), P("mp", None)),
        # self-attention
        ((r"self_attn", "(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        ((r"self_attn", "out_proj", "kernel"), P("mp", None)),
        # enc-dec attention
        ((r"encoder_attn", "(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        ((r"encoder_attn", "out_proj", "kernel"), P("mp", None)),
        # FFN
        ((r"fc1", "kernel"), P(None, "mp")),
        ((r"fc2", "kernel"), P("mp", None)),
        # layer norms
        ((r"layernorm_embedding", "(bias|scale)"), None),
        ((r"self_attn_layer_norm", "(bias|scale)"), None),
        ((r"encoder_attn_layer_norm", "(bias|scale)"), None),
        ((r"final_layer_norm", "(bias|scale)"), None),
        ((r"lm_head", "kernel"), P(None, "mp")),
    ]


def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print(k)
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))
