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

import os
import logging
import sys
import time
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
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.models.bart.modeling_flax_bart import BartConfig

import wandb

from dalle_mini.text import TextNormalizer
from dalle_mini.model import CustomFlaxBartForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    image_vocab_size: Optional[int] = field(
        default=None,
        metadata={"help": "Vocab size of image encoder"},
    )
    image_length: Optional[int] = field(
        default=None,
        metadata={"help": "Number of tokens per image"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name_or_path"
        },
    )
    normalize_text: bool = field(
        default=False,
        metadata={"help": "Whether to normalize text or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the authentication token for private datasets."
        },
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing. Not used in streaming mode."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets. Not used in streaming mode."
        },
    )

    def __post_init__(self):
        if self.dataset_repo_or_path is None:
            raise ValueError("Need a dataset repository or path.")


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to training parameters.
    """

    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate."}
    )
    adafactor: bool = field(
        default=False,
        metadata={"help": "Whether or not to replace AdamW by Adafactor."},
    )
    weight_decay: float = field(
        default=None, metadata={"help": "Weight decay if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm for Adafactor."}
    )
    use_decay: bool = field(
        default=False,
        metadata={"help": "Whether to use decay in the learning rate scheduler."},
    )

    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    logging_steps: int = field(
        default=40, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=400, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=4000, metadata={"help": "Save checkpoint every X updates steps."}
    )
    log_model: bool = field(
        default=False,
        metadata={"help": "Log model to wandb at `save_steps` frequency."},
    )

    seed_model: int = field(
        default=42,
        metadata={
            "help": "Random seed for the model that will be set at the beginning of training."
        },
    )
    # default seed of None ensures we don't repeat the same items if script was interrupted during an epoch
    seed_dataset: int = field(
        default=None,
        metadata={
            "help": "Random seed for the dataset that will be set at the beginning of training."
        },
    )

    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upload the trained model to the model hub after training."
        },
    )

    resume_from_wandb_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The reference to a wandb artifact for resuming training."},
    )


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )

    def restore_state(self, artifact_dir):
        # restore optimizer state
        with (Path(artifact_dir) / "opt_state.msgpack").open("rb") as f:
            new_opt_state = from_bytes(self.opt_state, f.read())

        # restore other parameters
        with (Path(artifact_dir) / "training_state.json").open("r") as f:
            training_state = json.load(f)

        # replace state
        return self.replace(
            opt_state=new_opt_state,
            step=training_state["step"],
            train_time=training_state["train_time"],
            train_samples=training_state["train_samples"],
        )


def data_loader(
    dataset: Dataset,
    batch_size: int,
    rng: jax.random.PRNGKey = None,
):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps_per_epoch = len(dataset) // batch_size

    if rng is not None:
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
    num_warmup_steps: int,
    learning_rate: float,
    use_decay: bool,
    num_train_steps: int = None,  # used only with `use_decay`, typically train_size // batch_size * num_epochs
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    if use_decay:
        assert (
            num_train_steps is not None
        ), "Learning rate with decay requires number of training steps"
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    if not use_decay:
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
    # See all possible arguments by passing the --help flag to this script.
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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load dataset
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }
    else:
        data_files = None
    dataset = load_dataset(
        data_args.dataset_repo_or_path,
        data_files=data_files,
        streaming=data_args.streaming,
        use_auth_token=data_args.use_auth_token,
    )

    # Set up wandb run
    wandb.init(
        entity="dalle-mini",
        project="dalle-mini",
        job_type="Seq2Seq",
        config=parser.parse_args(),
    )

    if training_args.resume_from_wandb_checkpoint is not None:
        artifact = wandb.run.use_artifact(training_args.resume_from_wandb_checkpoint)
        artifact_dir = artifact.download()

        # load model
        model = CustomFlaxBartForConditionalGeneration.from_pretrained(artifact_dir)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir,
            use_fast=True,
        )

    else:
        # Set up our new model config
        # TODO: simplify with custom config class
        config = BartConfig.from_pretrained(model_args.model_name_or_path)
        config.image_vocab_size = model_args.image_vocab_size
        config.image_length = model_args.image_length
        # we append decoder bos to image vocab
        config.decoder_start_token_id = config.image_vocab_size
        # ensure we don't generate bos (in addition to decoder start token)
        config.force_bos_token_to_be_generated = False
        config.forced_bos_token_id = None  # we don't need this token
        config.forced_eos_token_id = None  # we don't need this token

        config.tie_word_embeddings = False
        config.min_length = model_args.image_length + 1
        config.max_length = model_args.image_length + 1

        # below tokens need to be set to avoid error during generation (converted to jnp.array)
        # they are not expected to be used and are set to unreachable token id
        config.bos_token_id = config.image_vocab_size + 1
        config.pos_token_id = config.image_vocab_size + 1
        config.eos_token_id = config.image_vocab_size + 1

        # save whether we normalize the text
        config.normalize_text = model_args.normalize_text

        # Create a custom model and initialize it randomly
        model = CustomFlaxBartForConditionalGeneration(
            config, seed=training_args.seed_model, dtype=getattr(jnp, model_args.dtype)
        )

        # Load tokenizer
        if model_args.tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, use_fast=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
            )

    print(f"TPUs: {jax.device_count()}")
    assert jax.device_count() == 8, "TPUs in use, please check running processes"

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

    text_normalizer = TextNormalizer() if model.config.normalize_text else None

    def normalize_text(example):
        example[text_column] = text_normalizer(example[text_column])
        return example

    def preprocess_function(examples):
        inputs = examples[text_column]
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
            train_dataset = train_dataset.shuffle(1000, training_args.seed_dataset)
        else:
            seed_dataset = (
                training_args.seed_dataset
                if training_args.seed_dataset is not None
                else np.random.get_state()[1][0]
            )
            rng_dataset = jax.random.PRNGKey(seed_dataset)
        if model.config.normalize_text:
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
        if model.config.normalize_text:
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
    rng = jax.random.PRNGKey(training_args.seed_model)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    batch_size_per_update = train_batch_size * training_args.gradient_accumulation_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    len_train_dataset, len_eval_dataset = None, None
    if data_args.streaming:
        # we don't know the length, let's just assume max_samples if defined
        if data_args.max_train_samples is not None:
            len_train_dataset = data_args.max_train_samples
        if data_args.max_eval_samples is not None:
            len_eval_dataset = data_args.max_eval_samples
    else:
        len_train_dataset = len(train_dataset)
        len_eval_dataset = len(eval_dataset)
    steps_per_epoch = (
        len_train_dataset // train_batch_size if len_train_dataset is not None else None
    )
    num_train_steps = (
        steps_per_epoch * num_epochs if steps_per_epoch is not None else None
    )

    # Create learning rate schedule
    learning_rate_fn = create_learning_rate_fn(
        training_args.warmup_steps,
        training_args.learning_rate,
        training_args.use_decay,
        num_train_steps,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBart.
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
            weight_decay_rate=training_args.weight_decay,
            weight_decay_mask=decay_mask_fn,
            clipping_threshold=training_args.max_grad_norm,
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
    if training_args.resume_from_wandb_checkpoint is not None:
        # restore optimizer state and other parameters
        # we currently ignore partial epoch training: see https://github.com/borisdayma/dalle-mini/issues/105
        state = state.restore_state(artifact_dir)

    # label smoothed cross entropy
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # Define gradient update step fn
    def train_step(state, batch, delta_time):
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
        state = state.apply_gradients(
            grads=grads,
            dropout_rng=new_dropout_rng,
            train_time=state.train_time + delta_time,
            train_samples=state.train_samples + train_batch_size,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return state, metrics

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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & gradient accumulation) = {batch_size_per_update}"
    )
    epochs = tqdm(
        range(state.epoch, num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0
    )

    # set default x-axis as 'train/step'
    wandb_log({}, step=state.step)
    wandb.define_metric("*", step_metric="train/step")

    # add interesting config parameters
    wandb.config.update(
        {
            "len_train": len_train_dataset,
            "len_eval": len_eval_dataset,
            "batch_size_per_update": batch_size_per_update,
        }
    )

    # replicate state on each device
    state = state.replicate()

    def run_evaluation():
        # ======================== Evaluating ==============================
        eval_metrics = []
        if training_args.do_eval:
            if data_args.streaming:
                eval_loader = data_loader_streaming(eval_dataset, eval_batch_size)
            else:
                eval_loader = data_loader(eval_dataset, eval_batch_size)
            eval_steps = (
                len_eval_dataset // eval_batch_size
                if len_eval_dataset is not None
                else None
            )
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

    def run_save_model(state, eval_metrics=None):
        if jax.process_index() == 0:
            params = jax.device_get(unreplicate(state.params))
            # save model locally
            model.save_pretrained(
                training_args.output_dir,
                params=params,
            )

            # save tokenizer
            tokenizer.save_pretrained(training_args.output_dir)

            # save state
            opt_state = unreplicate(state.opt_state)
            with (Path(training_args.output_dir) / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(opt_state))
            state_dict = {
                k: jax.device_get(unreplicate(getattr(state, k))).item()
                for k in ["step", "epoch", "train_time", "train_samples"]
            }
            with (Path(training_args.output_dir) / "training_state.json").open(
                "w"
            ) as f:
                json.dump(
                    state_dict,
                    f,
                )

            # save to W&B
            if training_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("10GB"))

                metadata = dict(state_dict)
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics
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
                    commit_message=f"Saving weights and logs at step {unreplicate(state.step)+1}",
                    temp_dir=True,  # avoid issues with being in a repository
                )

    # init variables
    last_time = time.perf_counter()
    train_metric = None

    for epoch in epochs:
        state.replace(epoch=jax_utils.replicate(epoch))
        # ======================== Training ================================
        wandb_log({"train/epoch": epoch}, step=unreplicate(state.step))

        # Generate an epoch by shuffling sampling indices from the train dataset
        if data_args.streaming:
            train_dataset.set_epoch(epoch)  # shuffle dataset
            train_loader = data_loader_streaming(train_dataset, train_batch_size)
        else:
            rng_dataset, input_rng = jax.random.split(rng_dataset)
            train_loader = data_loader(train_dataset, train_batch_size, rng=input_rng)
        # train
        for batch in tqdm(
            train_loader,
            desc="Training...",
            position=1,
            leave=False,
            total=steps_per_epoch,
        ):

            # calculate delta time (we have a lag of one step but it's ok)
            new_time = time.perf_counter()
            delta_time = new_time - last_time
            last_time = new_time

            # train step
            state, train_metric = p_train_step(
                state, batch, jax_utils.replicate(delta_time)
            )
            step = unreplicate(state.step)

            if step % training_args.logging_steps == 0 and jax.process_index() == 0:
                # log metrics
                wandb_log(unreplicate(train_metric), step=step, prefix="train")
                # log state parameters
                state_dict = {
                    k.split("_")[-1]: unreplicate(getattr(state, k))
                    for k in ["epoch", "train_time", "train_samples"]
                }
                wandb_log(state_dict, step=step, prefix="train")

            eval_metrics = None
            if training_args.eval_steps and step % training_args.eval_steps == 0:
                eval_metrics = run_evaluation()

            if step % training_args.save_steps == 0:
                run_save_model(state, eval_metrics)

        # log final train metrics
        if train_metric is not None:
            train_metric = unreplicate(train_metric)
            wandb_log(train_metric, step=step, prefix="train")

            epochs.write(
                f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
            )

        # Final evaluation
        eval_metrics = run_evaluation()

        # save checkpoint after each epoch
        run_save_model(state, eval_metrics)


if __name__ == "__main__":
    main()
