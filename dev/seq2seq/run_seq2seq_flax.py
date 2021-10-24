#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for seq2seq, text to image.
Script adapted from run_summarization_flax.py
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import os
import logging as pylogging  # To avoid collision with transformers.utils.logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import json

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.serialization import from_bytes, to_bytes
import flax.linen as nn
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import (
    AutoTokenizer,
    FlaxBartForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.models.bart.modeling_flax_bart import *

import wandb

from dalle_mini.text import TextNormalizer

logger = pylogging.getLogger(__name__)


# Model hyperparameters, for convenience
# TODO: the model has now it's own definition file and should be imported
OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = "facebook/bart-large-cnn"  # we currently have issues with bart-large


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=BASE_MODEL,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )
    from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Loads a pretrained wandb checkpoint. Use artifact reference."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    text_column: Optional[str] = field(
        default="caption",
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    encoding_column: Optional[str] = field(
        default="encoding",
        metadata={
            "help": "The name of the column in the datasets containing the image encodings."
        },
    )
    dataset_repo_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset repository containing encoded files."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset."},
    )
    len_train: Optional[int] = field(
        default=None,
        metadata={"help": "Length of training dataset, required for streaming"},
    )
    len_eval: Optional[int] = field(
        default=None,
        metadata={"help": "Length of validation dataset, required for streaming"},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    no_decay: bool = field(
        default=False,
        metadata={"help": "Whether to use decay in the learning rate scheduler."},
    )
    max_target_length: Optional[int] = field(
        default=OUTPUT_LENGTH,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=OUTPUT_LENGTH,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the `max_length` param of `model.generate`, which is used "
            "during evaluation."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    normalize_text: bool = field(
        default=False,
        metadata={"help": "Normalize/Simplify text"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=80,  # ensure we have the same datasets cached data and avoid using too much space
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    log_interval: Optional[int] = field(
        default=40,
        metadata={"help": "Log frequency for metrics"},
    )
    log_model: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    save_model_steps: Optional[int] = field(
        default=5000,  # about once every 1.5h in our experiments
        metadata={
            "help": "For logging the model more frequently. Used only when `log_model` is set."
        },
    )

    def __post_init__(self):
        if self.dataset_repo_or_path is None:
            raise ValueError("Need a dataset repository or path.")
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "tsv",
                    "csv",
                    "json",
                    "jsonl",
                ], "`train_file` should be a tsv, csv or json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "tsv",
                    "csv",
                    "json",
                    "jsonl",
                ], "`validation_file` should be a tsv, csv or json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.streaming and (self.len_train is None or self.len_eval is None):
            raise ValueError(
                "Streaming requires providing length of training and validation datasets"
            )


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray = None

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )

    def restore_state(self, artifact_dir):
        # restore optimizer state
        with (Path(artifact_dir) / "opt_state.msgpack").open("rb") as f:
            opt_state = from_bytes(self.opt_state, f.read())

        # restore steps
        with (Path(artifact_dir) / "training_state.json").open("r") as f:
            training_state = json.load(f)
        step = training_state["step"]

        # replace state
        return self.replace(step=step, opt_state=opt_state)


class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # check config is valid, otherwise set default values
        self.config.vocab_size_output = getattr(
            self.config, "vocab_size_output", OUTPUT_VOCAB_SIZE
        )
        self.config.max_position_embeddings_decoder = getattr(
            self.config, "max_position_embeddings_decoder", OUTPUT_LENGTH
        )

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
        self.encoder = FlaxBartEncoder(
            self.config, dtype=self.dtype, embed_tokens=self.shared
        )

        # the decoder has a different config
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = (
            self.config.max_position_embeddings_decoder
        )
        decoder_config.vocab_size = self.config.vocab_size_output
        self.decoder = FlaxBartDecoder(
            decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed
        )


class CustomFlaxBartForConditionalGenerationModule(
    FlaxBartForConditionalGenerationModule
):
    def setup(self):
        # check config is valid, otherwise set default values
        self.config.vocab_size_output = getattr(
            self.config, "vocab_size_output", OUTPUT_VOCAB_SIZE
        )

        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size_output,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.final_logits_bias = self.param(
            "final_logits_bias", self.bias_init, (1, self.config.vocab_size_output)
        )


class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule


def data_loader(
    rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False
):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)
        yield batch


def data_loader_streaming(dataset: Dataset, batch_size: int):
    keys = ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
    batch = {k: [] for k in keys}
    for item in dataset:
        for k, v in item.items():
            batch[k].append(v)
        if len(batch[keys[0]]) == batch_size:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            batch = shard(batch)
            yield batch
            batch = {k: [] for k in keys}


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    no_decay: bool,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    if no_decay:
        return warmup_fn
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def wandb_log(metrics, step=None, prefix=None):
    if jax.process_index() == 0:
        log_metrics = {
            f"{prefix}/{k}" if prefix is not None else k: v for k, v in metrics.items()
        }
        if step is not None:
            log_metrics["train/step"] = step
        wandb.log(log_metrics)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    pylogging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=pylogging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(pylogging.INFO if jax.process_index() == 0 else pylogging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }
    dataset = load_dataset(
        data_args.dataset_repo_or_path,
        data_files=data_files,
        streaming=data_args.streaming,
    )

    # Set up wandb run
    wandb.init(
        entity="dalle-mini",
        project="dalle-mini",
        job_type="Seq2Seq",
        config=parser.parse_args(),
        save_code=True,
    )

    if model_args.from_checkpoint is not None:
        artifact = wandb.run.use_artifact(model_args.from_checkpoint)
        artifact_dir = artifact.download()
        model = CustomFlaxBartForConditionalGeneration.from_pretrained(artifact_dir)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir,
            use_fast=model_args.use_fast_tokenizer,
        )

    else:
        # Set up our new model config
        config = BartConfig.from_pretrained(model_args.model_name_or_path)
        config.tie_word_embeddings = False
        config.decoder_start_token_id = BOS_TOKEN_ID  # for first token
        config.bos_token_id = (
            BOS_TOKEN_ID  # should not be used (due to forced_bos_token_id)
        )
        config.pos_token_id = (
            BOS_TOKEN_ID  # should not be needed (as we generate until max_length)
        )
        config.eos_token_id = BOS_TOKEN_ID + 1  # unreachable
        config.forced_bos_token_id = None  # we don't need this token
        config.forced_eos_token_id = None  # we don't need this token
        config.force_bos_token_to_be_generated = (
            False  # otherwise it sets bos_token_id at loading
        )
        config.min_length = data_args.max_target_length
        config.max_length = data_args.max_target_length

        # Create a custom model and initialize it randomly
        model = CustomFlaxBartForConditionalGeneration(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
        )

    print(f"TPUs: {jax.device_count()}")
    assert jax.device_count() == 8, "TPUs in use, please check running processes"

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    # Get the column names for input/target.
    text_column = data_args.text_column
    encoding_column = data_args.encoding_column

    def shift_tokens_right(input_ids: np.array, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id
        return shifted_input_ids

    text_normalizer = TextNormalizer() if data_args.normalize_text else None

    def normalize_text(example):
        example[text_column] = text_normalizer(example[text_column])
        return example

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs] if prefix else inputs
        # Setting padding="max_length" as we need fixed length inputs for jitted functions
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # set up targets
        # Note: labels correspond to our target indices
        # decoder input ids are the same but shifted to the right with bos at the beginning (and without last token)
        labels = examples[encoding_column]
        labels = np.asarray(labels)

        # We need the labels, in addition to the decoder_input_ids, for the compute_loss function
        model_inputs["labels"] = labels

        # In our case, this prepends the bos token and removes the last one
        decoder_input_ids = shift_tokens_right(
            labels, model.config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = (
                train_dataset.take(data_args.max_train_samples)
                if data_args.streaming
                else train_dataset.select(range(data_args.max_train_samples))
            )
        if data_args.streaming:
            train_dataset = train_dataset.shuffle(1000, training_args.seed)
        if data_args.normalize_text:
            train_dataset = (
                train_dataset.map(normalize_text)
                if data_args.streaming
                else train_dataset.map(
                    normalize_text,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Normalizing the validation dataset",
                )
            )
        train_dataset = (
            train_dataset.map(
                preprocess_function,
                batched=True,
            )
            if data_args.streaming
            else train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = (
                eval_dataset.take(data_args.max_train_samples)
                if data_args.streaming
                else eval_dataset.select(range(data_args.max_train_samples))
            )
        if data_args.normalize_text:
            eval_dataset = (
                eval_dataset.map(normalize_text)
                if data_args.streaming
                else eval_dataset.map(
                    normalize_text,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Normalizing the validation dataset",
                )
            )
        eval_dataset = (
            eval_dataset.map(
                preprocess_function,
                batched=True,
            )
            if data_args.streaming
            else eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    batch_size_per_update = train_batch_size * training_args.gradient_accumulation_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    if data_args.streaming:
        len_train_dataset = data_args.len_train
        if (
            data_args.max_train_samples is not None
            and data_args.max_train_samples < len_train_dataset
        ):
            len_train_dataset = data_args.max_train_samples

        len_eval_dataset = data_args.len_eval
        if (
            data_args.max_eval_samples is not None
            and data_args.max_eval_samples < len_eval_dataset
        ):
            len_eval_dataset = data_args.max_eval_samples
    else:
        len_train_dataset = len(train_dataset)
        len_eval_dataset = len(eval_dataset)
    steps_per_epoch = len_train_dataset // train_batch_size
    total_steps = steps_per_epoch * num_epochs
    total_optimization_steps = (len_train_dataset // batch_size_per_update) * num_epochs

    # Create learning rate schedule
    learning_rate_fn = create_learning_rate_fn(
        len_train_dataset,
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
        data_args.no_decay,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBart.
    # For FlaxT5, one should correct the layer norm parameter naming
    # accordingly - see `run_t5_mlm_flax.py` e.g.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_params = [
            (name, "scale")
            for name in [
                "self_attn_layer_norm",
                "layernorm_embedding",
                "final_layer_norm",
            ]
        ]
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in layer_norm_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=learning_rate_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # add gradient accumulation
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.chain(
            optax.apply_every(training_args.gradient_accumulation_steps), optimizer
        )

    # Setup train state
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )
    if model_args.from_checkpoint is not None:
        # restore optimizer state and step
        state = state.restore_state(artifact_dir)
        # TODO: number of remaining training epochs/steps and dataloader state need to be adjusted

    # label smoothed cross entropy
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params, batch):
            labels = batch.pop("labels")
            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params, batch)
        grads = jax.lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return state.replace(dropout_rng=new_dropout_rng), metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & gradient accumulation) = {batch_size_per_update}"
    )
    logger.info(f"  Total global steps = {total_steps}")
    logger.info(f"  Total optimization steps = {total_optimization_steps}")

    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

    # set default x-axis as 'train/step'
    wandb_log({}, step=unreplicate(state.step))
    wandb.define_metric("*", step_metric="train/step")

    # add interesting config parameters
    wandb.config.update(
        {
            "len_train": len_train_dataset,
            "len_eval": len_eval_dataset,
            "batch_size_per_update": batch_size_per_update,
            "total_steps": total_steps,
            "total_optimization_steps": total_optimization_steps,
        }
    )

    def run_evaluation():
        # ======================== Evaluating ==============================
        eval_metrics = []
        if training_args.do_eval:
            if data_args.streaming:
                eval_loader = data_loader_streaming(eval_dataset, eval_batch_size)
            else:
                eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size)
            eval_steps = len_eval_dataset // eval_batch_size
            for batch in tqdm(
                eval_loader,
                desc="Evaluating...",
                position=2,
                leave=False,
                total=eval_steps,
            ):
                # Model forward
                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # log metrics
            wandb_log(eval_metrics, step=unreplicate(state.step), prefix="eval")

            # Print metrics and update progress bar
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

            return eval_metrics

    def run_save_model(state, step, epoch, eval_metrics=None):
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))

            # save model locally
            model.save_pretrained(
                training_args.output_dir,
                params=params,
            )

            # save tokenizer
            tokenizer.save_pretrained(training_args.output_dir)

            # save state
            # TODO: maybe we should just save the full state object without params
            state = unreplicate(state)
            with (Path(training_args.output_dir) / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(state.opt_state))
            with (Path(training_args.output_dir) / "training_state.json").open(
                "w"
            ) as f:
                json.dump({"step": state.step.item()}, f)

            # save to W&B
            if data_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("5GB"))

                metadata = {"step": step, "epoch": epoch}
                if eval_metrics is not None:
                    metadata["eval/loss"] = eval_metrics["loss"]
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", type="bart_model", metadata=metadata
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "flax_model.msgpack")
                )
                artifact.add_file(str(Path(training_args.output_dir) / "config.json"))
                artifact.add_file(
                    str(Path(training_args.output_dir) / "tokenizer.json")
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "tokenizer_config.json")
                )
                artifact.add_file(str(Path(training_args.output_dir) / "vocab.json"))
                artifact.add_file(str(Path(training_args.output_dir) / "merges.txt"))
                artifact.add_file(
                    str(Path(training_args.output_dir) / "special_tokens_map.json")
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "opt_state.msgpack")
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "training_state.json")
                )

                wandb.run.log_artifact(artifact)

            # save to the hub
            if training_args.push_to_hub:
                model.save_pretrained(
                    training_args.output_dir,
                    params=params,
                    push_to_hub=training_args.push_to_hub,
                    commit_message=f"Saving weights and logs of epoch {epoch+1}",
                    temp_dir=True,  # avoid issues with being in a repository
                )

    for epoch in epochs:
        # ======================== Training ================================
        step = unreplicate(state.step)
        # wandb_log({"train/epoch": epoch}, step=step)

        # Generate an epoch by shuffling sampling indices from the train dataset
        if data_args.streaming:
            train_dataset.set_epoch(epoch)
            train_loader = data_loader_streaming(train_dataset, train_batch_size)
        else:
            rng, input_rng = jax.random.split(rng)
            train_loader = data_loader(
                input_rng, train_dataset, train_batch_size, shuffle=True
            )
        # train
        for batch in tqdm(
            train_loader,
            desc="Training...",
            position=1,
            leave=False,
            total=steps_per_epoch,
        ):
            state, train_metric = p_train_step(state, batch)
            step = unreplicate(state.step)

            if step % data_args.log_interval == 0 and jax.process_index() == 0:
                # log metrics
                wandb_log(unreplicate(train_metric), step=step, prefix="train")

            if training_args.eval_steps and step % training_args.eval_steps == 0:
                run_evaluation()

            if step % data_args.save_model_steps == 0:
                run_save_model(state, step, epoch)

        # log final train metrics
        wandb_log(unreplicate(train_metric), step=step, prefix="train")

        train_metric = unreplicate(train_metric)
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
        )

        # Final evaluation
        eval_metrics = run_evaluation()

        # save checkpoint after each epoch
        run_save_model(state, state.step, epoch, eval_metrics)


if __name__ == "__main__":
    main()
