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
        (("embed_positions", "embedding"), P("mp", None)),
        (("embed_tokens", "embedding"), P("mp", None)),
        (("rel_bias", "embedding"), P(None, "mp")),
        # attention
        (("(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        (("out_proj", "kernel"), P("mp", None)),
        # FFN
        (("Dense_0", "kernel"), P(None, "mp")),
        (("GLU.*", "Dense_1", "kernel"), P(None, "mp")),
        (("GLU.*", "Dense_2", "kernel"), P("mp", None)),
        (("FFN.*", "Dense_1", "kernel"), P("mp", None)),
        # layer norms
        (("(bias|scale)",), None),
        (("lm_head", "kernel"), P(None, "mp")),
        # head scale and tau
        (("(head_scale|tau)",), None),
    ]


def set_partitions(in_dict, use_scan):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print(f"Unmatched -> {k}")
    l = list(result.keys())
    if use_scan:
        # add None dimension to layers
        result = {
            k: (P(*(None,) + v) if v is not None else None)
            if any(x in k for x in ["FlaxBartEncoderLayers", "FlaxBartDecoderLayers"])
            else v
            for k, v in result.items()
        }
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))
