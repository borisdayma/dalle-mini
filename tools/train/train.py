#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-2022 The HuggingFace & DALL·E Mini team. All rights reserved.
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
Training DALL·E Mini.
Script adapted from run_summarization_flax.py
"""

import copy
import io
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
import wandb
from datasets import Dataset
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import onehot
from google.cloud import storage
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from scalable_shampoo.distributed_shampoo import GraftingType, distributed_shampoo
from tqdm import tqdm
from transformers import HfArgumentParser

from dalle_mini.data import Dataset
from dalle_mini.model import (
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer,
    set_partitions,
)

cc.initialize_cache("./jax_cache", max_cache_size_bytes=10 * 2**30)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Don't set if you want to train a model from scratch. "
            "W&B artifact references are supported in addition to the sources supported by `PreTrainedModel`."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name_or_path"
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the computations will be performed (not the model weights). Choose one of `[float32, float16, bfloat16]`."
        },
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Restore optimizer and training state. Can be True (will retrieve associated wandb artifact), a local directory or a Google bucket path."
        },
    )

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
            assert (
                self.tokenizer_name is not None
            ), "Tokenizer name or model name/path needs to be specified"
        if self.restore_state:
            assert self.model_name_or_path is not None and (
                "/model-" in self.model_name_or_path
            ), "Restoring state only available with W&B artifact reference"

    def get_metadata(self):
        if self.restore_state:
            if jax.process_index() == 0:
                artifact = wandb.run.use_artifact(self.model_name_or_path)
            else:
                artifact = wandb.Api().artifact(self.model_name_or_path)
            return artifact.metadata
        else:
            return dict()

    def get_opt_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if self.restore_state is True:
                # wandb artifact
                state_artifact = self.model_name_or_path.replace(
                    "/model-", "/state-", 1
                )
                if jax.process_index() == 0:
                    artifact = wandb.run.use_artifact(state_artifact)
                else:
                    artifact = wandb.Api().artifact(state_artifact)
                if artifact.metadata.get("bucket_path"):
                    # we will read directly file contents
                    self.restore_state = artifact.metadata["bucket_path"]
                else:
                    artifact_dir = artifact.download(tmp_dir)
                    self.restore_state = str(Path(artifact_dir) / "opt_state.msgpack")

            if self.restore_state.startswith("gs://"):
                bucket_path = Path(self.restore_state[5:]) / "opt_state.msgpack"
                bucket, blob_name = str(bucket_path).split("/", 1)
                client = storage.Client()
                bucket = client.bucket(bucket)
                blob = bucket.blob(blob_name)
                return blob.download_as_bytes()

            with Path(self.restore_state).open("rb") as f:
                return f.read()


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
    dataset_repo_or_path: str = field(
        default=None,
        metadata={"help": "The dataset repository containing encoded files."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data file (glob & braceexpand acceptable)."
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file (glob & braceexpand acceptable)."
        },
    )
    # data loading should not be a bottleneck so we use "streaming" mode by default
    streaming: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to stream the dataset."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use the authentication token for private datasets."
        },
    )
    shard_by_host: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to shard data files by host in multi-host environments."
        },
    )
    blank_caption_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of removing some captions for classifier-free guidance."
        },
    )
    clip_score_column: Optional[str] = field(
        default="clip_score",
        metadata={"help": "Column that containts clip score for filtering."},
    )
    min_clip_score: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum clip score required."},
    )
    max_clip_score: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum clip score required."},
    )
    filter_column: Optional[str] = field(
        default=None,
        metadata={"help": "Column that containts classes to be filtered."},
    )
    filter_value: Optional[str] = field(
        default=None,
        metadata={"help": "Class value to be kept during filtering."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples."
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
    # default seed of None ensures we don't repeat the same items if script was interrupted during an epoch
    seed_dataset: int = field(
        default=None,
        metadata={
            "help": "Random seed for the dataset that will be set at the beginning of training."
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
        default=False, metadata={"help": "Whether to run eval on the validation set."}
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per data parallel device for training."},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size per data parallel device for evaluation. Same as training batch size if not set."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing an update pass."
        },
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Use gradient checkpointing."}
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate."}
    )
    optim: str = field(
        default="distributed_shampoo",
        metadata={
            "help": 'The optimizer to use. Can be "distributed_shampoo" (default), "adam" or "adafactor"'
        },
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm for Adafactor."}
    )
    block_size: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
    )
    preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": "The type of grafting to use. Can be 'rmsprop_normalized' (default), 'rmsprop', 'adagrad', 'adagrad_normalized', 'sgd' or 'sqrt_n'"
        },
    )
    optim_quantized: bool = field(
        default=False,
        metadata={
            "help": "Whether to quantize optimizer (only supported with Distributed Shampoo)."
        },
    )

    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )

    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    lr_decay: str = field(
        default=None,
        metadata={
            "help": "Decay to be used in the learning rate scheduler. Can be None (default), linear or exponential."
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": "Number of transition steps associated with learning rate decay when using exponential decay."
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={
            "help": "Decay rate associated with learning rate when using exponential decay."
        },
    )
    lr_staircase: bool = field(
        default=False,
        metadata={
            "help": "Whether to use staircase or continuous learning rate when using exponential decay."
        },
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
    log_histograms: bool = field(
        default=False,
        metadata={
            "help": "Log parameters and gradients histograms. Slows down training."
        },
    )

    seed_model: int = field(
        default=42,
        metadata={
            "help": "Random seed for the model that will be set at the beginning of training."
        },
    )

    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb entity to use (for teams)."},
    )
    wandb_project: str = field(
        default="dalle-mini",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="Seq2Seq",
        metadata={"help": "The name of the wandb job type."},
    )

    assert_TPU_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )

    mp_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of devices required for model parallelism. The other dimension of available devices is used for data parallelism."
        },
    )

    dp_devices: int = field(init=False)

    def __post_init__(self):
        if self.assert_TPU_available:
            assert (
                jax.local_device_count() == 8
            ), "TPUs in use, please check running processes"
        assert self.optim in [
            "distributed_shampoo",
            "adam",
            "adafactor",
        ], f"Selected optimizer not supported: {self.optim}"
        assert self.graft_type in [
            "rmsprop_normalized",
            "rmsprop",
            "adagrad",
            "adagrad_normalized",
            "sgd",
            "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"
        assert self.lr_decay in [
            None,
            "linear",
            "exponential",
        ], f"Selected learning rate decay not supported: {self.lr_decay}"
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and self.do_train
            and not self.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        assert (
            self.mp_devices > 0
        ), f"Number of devices for model parallelism must be > 0"
        assert (
            jax.device_count() % self.mp_devices == 0
        ), f"Number of available devices ({jax.device_count()} must be divisible by number of devices used for model parallelism ({self.mp_devices})."
        self.dp_devices = jax.device_count() // self.mp_devices


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen


class MetricsLogger:
    def __init__(self, step):
        self.step = step
        self.time = time.perf_counter()
        self.state_dict = {}

    def update_state_metrics(self, state):
        """Update internal state metrics (logged at each call to be used as x-axis)"""
        self.state_dict = {
            f'train/{k.split("_")[-1]}': getattr(state, k)
            for k in ["step", "epoch", "train_time", "train_samples"]
        }
        # timing metrics
        new_step = int(state.step)
        new_time = time.perf_counter()
        if new_step > self.step:
            time_per_step = (new_time - self.time) / (new_step - self.step)
            self.step = new_step
            self.time = new_time
            self.state_dict["train/time_per_step"] = time_per_step

    def log(self, metrics, prefix=None):
        if jax.process_index() == 0:
            log_metrics = {}
            for k, v in metrics.items():
                if prefix is not None:
                    k = f"{prefix}/{k}"
                if "_norm" in k:
                    log_metrics[f"{k}/"] = unfreeze(v)
                elif "_hist" in k:
                    v = jax.tree_map(lambda x: jax.device_get(x), unfreeze(v))
                    v = jax.tree_map(
                        lambda x: wandb.Histogram(np_histogram=x),
                        v,
                        is_leaf=lambda x: isinstance(x, tuple),
                    )
                    log_metrics[f"{k}/"] = v
                else:
                    log_metrics[k] = v
            wandb.log({**log_metrics, **self.state_dict})


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
    dataset = Dataset(
        **asdict(data_args),
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )

    logger.info(f"Local TPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs: {jax.device_count()}")

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=parser.parse_args(),
        )

    # Set up our new model config
    if model_args.config_name:
        config = DalleBartConfig.from_pretrained(model_args.config_name)
        # initializing params with gradient checkpointing creates issues
        # we correctly set it later per training_args
        config.gradient_checkpointing = False
    else:
        config = None

    # Load or create new model
    if model_args.model_name_or_path:
        model = DalleBart.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            abstract_init=True,  # we overwrite them with loaded checkpoint
            # initializing params with gradient checkpointing creates issues
            # we correctly set it later per training_args
            gradient_checkpointing=False,
        )
    else:
        model = DalleBart(
            config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            abstract_init=True,
        )

    # define model eval and train functions
    eval_fn = model.__call__
    if training_args.gradient_checkpointing:
        remat_config = copy.deepcopy(model.config)
        remat_config.gradient_checkpointing = True
        remat_model = DalleBart(
            remat_config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            init_weights=False,
        )
        train_fn = remat_model.__call__
    else:
        train_fn = model.__call__

    # get model metadata
    model_metadata = model_args.get_metadata()

    # get PartitionSpec for model params (required to be a dict)
    param_spec = set_partitions(model.params)

    # convert params to frozen dict
    model._params = freeze(model.params)

    # Load tokenizer
    tokenizer = DalleBartTokenizer.from_pretrained(
        model_args.tokenizer_name, use_fast=True
    )

    # Preprocessing the datasets.
    # We need to normalize and tokenize inputs and targets.
    dataset.preprocess(tokenizer=tokenizer, config=model.config)

    # Initialize our training
    dropout_rng = jax.random.PRNGKey(training_args.seed_model)

    # Store some constant
    num_epochs = training_args.num_train_epochs
    # batch size
    batch_size_per_node_per_grad_step = (
        training_args.per_device_train_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
    )
    batch_size_per_node = (
        batch_size_per_node_per_grad_step * training_args.gradient_accumulation_steps
    )
    batch_size_per_step = batch_size_per_node * jax.process_count()
    eval_batch_size_per_node = (
        training_args.per_device_eval_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
    )
    eval_batch_size_per_step = eval_batch_size_per_node * jax.process_count()
    len_train_dataset, len_eval_dataset = dataset.length
    steps_per_epoch = (
        len_train_dataset // batch_size_per_node
        if len_train_dataset is not None
        else None
    )
    num_train_steps = (
        steps_per_epoch * num_epochs if steps_per_epoch is not None else None
    )
    num_params = model.num_params

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Batch size per dp device = {training_args.per_device_train_batch_size}"
    )
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(
        f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Batch size per update = {batch_size_per_step}")
    logger.info(f"  Model parameters = {num_params:,}")

    # set up wandb run
    if jax.process_index() == 0:
        # set default x-axis as 'train/step'
        wandb.define_metric("*", step_metric="train/step")

        # add interesting config parameters
        wandb.config.update(
            {
                "len_train_dataset": len_train_dataset,
                "len_eval_dataset": len_eval_dataset,
                "batch_size_per_step": batch_size_per_step,
                "num_params": num_params,
                "num_devices": jax.device_count(),
            }
        )

    # Create learning rate schedule
    def create_learning_rate_fn() -> Callable[[int], jnp.array]:
        """Create the learning rate function."""
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=training_args.learning_rate,
            transition_steps=training_args.warmup_steps,
        )
        # offset step when resuming
        if model_metadata.get("step", 0):
            warmup_fn = optax.join_schedules(
                schedules=[optax.constant_schedule(0.0), warmup_fn],
                boundaries=[model_metadata["step"]],
            )
        if training_args.lr_decay is None:
            return warmup_fn
        elif training_args.lr_decay == "linear":
            assert (
                num_train_steps is not None
            ), "linear decay requires knowing the dataset length"
            decay_fn = optax.linear_schedule(
                init_value=training_args.learning_rate,
                end_value=0,
                transition_steps=num_train_steps - training_args.warmup_steps,
            )
        elif training_args.lr_decay == "exponential":
            decay_fn = optax.exponential_decay(
                init_value=training_args.learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[model_metadata.get("step", 0) + training_args.warmup_steps],
        )
        return schedule_fn

    learning_rate_fn = create_learning_rate_fn()

    # create adam optimizer
    if training_args.optim == "distributed_shampoo":
        # parameters from https://github.com/tensorflow/lingvo/blob/03ee9d7cd50764b0424c7c863733c91fc0b053ec/lingvo/jax/optimizers.py#L729
        graft_type = {
            "sgd": GraftingType.SGD,
            "adagrad": GraftingType.ADAGRAD,
            "rmsprop": GraftingType.RMSPROP,
            "rmsprop_normalized": GraftingType.RMSPROP_NORMALIZED,
            "sqrt_n": GraftingType.SQRT_N,
            "adagrad_normalized": GraftingType.ADAGRAD_NORMALIZED,
        }[training_args.graft_type]
        optimizer = distributed_shampoo(
            learning_rate_fn,
            block_size=training_args.block_size,
            beta1=training_args.beta1,
            beta2=training_args.beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=1e-6,
            start_preconditioning_step=max(
                training_args.preconditioning_compute_steps + 1, 101
            ),
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=graft_type,
            nesterov=False,
            exponent_override=0,
            statistics_partition_spec=PartitionSpec(None, "dp", None),
            preconditioner_partition_spec=PartitionSpec("dp", None, None),
            num_devices_for_pjit=training_args.dp_devices,
            shard_optimizer_states=True,
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=True,
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=None,
            precision=jax.lax.Precision.HIGHEST,
            best_effort_memory_usage_reduction=training_args.optim_quantized,
        )
        # get the real optimizer and helper functions
        update_fn = optimizer.update
        optimizer = optimizer.init(model.params)
        opt_fn = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
            optimizer.pspec_fn, optimizer.shape_and_dtype_fn
        )
        optimizer = optax.GradientTransformation(optimizer.init_fn, update_fn)

    elif training_args.optim == "adam":
        optimizer = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=training_args.beta1,
            b2=training_args.beta2,
            eps=training_args.adam_epsilon,
        )
    elif training_args.optim == "adafactor":
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=learning_rate_fn,
            clipping_threshold=training_args.max_grad_norm,
        )

    # get PartitionSpec for optimizer state
    def get_opt_state_spec_and_shape(param_spec):
        # get opt_state shape without actual init
        opt_state_shape = jax.eval_shape(optimizer.init, model.params)

        if training_args.optim == "adam":

            def _opt_state_spec_per_leaf(x):
                if isinstance(x, FrozenDict):
                    # variables with same structure as params
                    return param_spec
                else:
                    # other variables such as count
                    return None

            opt_state_spec = jax.tree_map(
                _opt_state_spec_per_leaf,
                opt_state_shape,
                # return None spec for empty elements
                is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
            )

        elif training_args.optim == "adafactor":
            # factorized state must be replicated (rank different than params)
            opt_state_spec = None

        elif training_args.optim == "distributed_shampoo":
            opt_state_spec = opt_fn.pspec_fn(
                params=model.params,
                params_partition_spec=param_spec,
                partition_spec_for_statistics=PartitionSpec(None, "dp", None),
            )
        else:
            raise NotImplementedError
        return opt_state_spec, opt_state_shape

    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape(param_spec)

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))
    logger.info(f"  Mesh shape: {mesh_shape}")

    # define state spec
    state_spec = TrainState(
        params=param_spec,
        opt_state=opt_state_spec,
        dropout_rng=None,
        step=None,
        epoch=None,
        train_time=None,
        train_samples=None,
        apply_fn=train_fn,
        tx=optimizer,
    )

    # init params if not available yet
    def maybe_init_params(params):
        if model_args.model_name_or_path:
            # model params are correctly loaded
            return params
        else:
            # params have not been initialized yet
            return model.init_weights()

    with maps.mesh(mesh.devices, mesh.axis_names):
        logger.info("  Creating state")
        if not model_args.restore_state:

            def init_state(params):
                return TrainState.create(
                    apply_fn=train_fn,
                    tx=optimizer,
                    params=maybe_init_params(params),
                    dropout_rng=dropout_rng,
                )

            state = pjit(
                init_state,
                in_axis_resources=(param_spec,)
                if model_args.model_name_or_path
                else None,
                out_axis_resources=state_spec,
                donate_argnums=(0,),
            )(model.params if model_args.model_name_or_path else None)

        else:
            # load opt_state
            opt_state = from_bytes(opt_state_shape, model_args.get_opt_state())

            # restore other attributes
            attr_state = {
                k: model_metadata[k]
                for k in ["step", "epoch", "train_time", "train_samples"]
            }

            def restore_state(params, opt_state):
                return TrainState(
                    apply_fn=train_fn,
                    tx=optimizer,
                    params=params,
                    opt_state=opt_state,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                restore_state,
                in_axis_resources=(
                    param_spec,
                    opt_state_spec,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(model.params, opt_state)

            # remove opt_state from CPU
            del opt_state

    # free CPU memory
    del model._params, opt_state_spec, opt_state_shape

    # define batch specs
    batch_spec = PartitionSpec("dp")
    grad_batch_spec = PartitionSpec(None, "dp")

    # define loss
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # "vmap trick" avoids a crash when mp_devices > 1 (not sure why it happens)
    # lead to better perf: see https://wandb.ai/dalle-mini/dalle-mini/reports/JAX-pmap-vs-pjit--VmlldzoxNDg1ODA2
    use_vmap_trick = True

    # make grad_param_spec for vmap
    if use_vmap_trick:
        grad_param_spec = jax.tree_map(
            lambda x: PartitionSpec(*("dp",) + (x if x is not None else (None,))),
            param_spec,
        )

    # Define gradient update step fn
    def train_step(state, batch, delta_time):

        # get a minibatch (one gradient accumulation slice)
        def get_minibatch(batch, grad_idx):
            return jax.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch,
            )

        def compute_loss(params, minibatch, dropout_rng):
            # minibatch has dim (batch_size, ...)
            minibatch, labels = minibatch.pop("labels")
            logits = state.apply_fn(
                **minibatch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            return loss_fn(logits, labels)

        grad_fn = jax.value_and_grad(compute_loss)

        def loss_and_grad(grad_idx, dropout_rng):
            # minibatch at grad_idx for gradient accumulation (None otherwise)
            minibatch = (
                get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            )
            # ensure it is sharded properly
            minibatch = with_sharding_constraint(minibatch, batch_spec)
            # only 1 single rng per grad step, let us handle larger batch size (not sure why)
            dropout_rng, _ = jax.random.split(dropout_rng)

            if use_vmap_trick:
                # "vmap trick", calculate loss and grads independently per dp_device
                loss, grads = jax.vmap(
                    grad_fn, in_axes=(None, 0, None), out_axes=(0, 0)
                )(state.params, minibatch, dropout_rng)
                # ensure they are sharded correctly
                loss = with_sharding_constraint(loss, batch_spec)
                grads = with_sharding_constraint(grads, grad_param_spec)
                # average across all devices
                # Note: we could average per device only after gradient accumulation, right before params update
                loss, grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), (loss, grads))
            else:
                # "vmap trick" does not work in multi-hosts and requires too much hbm
                loss, grads = grad_fn(state.params, minibatch, dropout_rng)
            # ensure grads are sharded
            grads = with_sharding_constraint(grads, param_spec)
            # return loss and grads
            return loss, grads, dropout_rng

        if training_args.gradient_accumulation_steps == 1:
            loss, grads, dropout_rng = loss_and_grad(None, state.dropout_rng)
        else:
            # create initial state for cumul_minibatch_step loop
            init_minibatch_step = (
                0.0,
                with_sharding_constraint(
                    jax.tree_map(jnp.zeros_like, state.params), param_spec
                ),
                state.dropout_rng,
            )

            # accumulate gradients
            def cumul_minibatch_step(grad_idx, cumul_loss_grad_dropout):
                cumul_loss, cumul_grads, dropout_rng = cumul_loss_grad_dropout
                loss, grads, dropout_rng = loss_and_grad(grad_idx, dropout_rng)
                cumul_loss, cumul_grads = jax.tree_map(
                    jnp.add, (cumul_loss, cumul_grads), (loss, grads)
                )
                cumul_grads = with_sharding_constraint(cumul_grads, param_spec)
                return cumul_loss, cumul_grads, dropout_rng

            # loop over gradients
            loss, grads, dropout_rng = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_minibatch_step,
            )
            grads = with_sharding_constraint(grads, param_spec)
            # sum -> mean
            loss, grads = jax.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps, (loss, grads)
            )

        grads = with_sharding_constraint(grads, param_spec)

        # update state
        state = state.apply_gradients(
            grads=grads,
            dropout_rng=dropout_rng,
            train_time=state.train_time + delta_time,
            train_samples=state.train_samples + batch_size_per_step,
        )

        # get norm and histogram of grads and params
        zeros_norm = jax.tree_map(lambda _: jnp.float32(0), state.params)

        def maybe_fn(fn, val, zeros):
            """Call fn only if it is a logging step"""
            return jax.lax.cond(
                state.step % training_args.logging_steps == 0,
                fn,
                lambda _: zeros,
                val,
            )

        def norm(val):
            return jax.tree_map(lambda x: jnp.linalg.norm(x), val)

        gradients_norm = maybe_fn(norm, grads, zeros_norm)
        params_norm = maybe_fn(norm, state.params, zeros_norm)

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
            "gradients_norm": gradients_norm,
            "params_norm": params_norm,
        }

        if training_args.log_histograms:
            zeros_hist = jax.tree_map(
                lambda _: jnp.histogram(jnp.zeros(1), density=True), state.params
            )

            def histogram(val):
                return jax.tree_map(lambda x: jnp.histogram(x, density=True), val)

            gradients_hist = maybe_fn(histogram, grads, zeros_hist)
            params_hist = maybe_fn(histogram, state.params, zeros_hist)

            metrics.update(
                {
                    "params_hist": params_hist,
                    "gradients_hist": gradients_hist,
                }
            )

        return state, metrics

    # Define eval fn
    def eval_step(state, batch):
        def compute_eval_loss(batch):
            batch, labels = batch.pop("labels")
            logits = eval_fn(**batch, params=state.params, train=False)[0]
            return loss_fn(logits, labels)

        if use_vmap_trick:
            loss = jax.vmap(compute_eval_loss)(batch)
            # ensure they are sharded correctly
            loss = with_sharding_constraint(loss, batch_spec)
            # average across all devices
            loss = jnp.mean(loss)
        else:
            loss = compute_eval_loss(batch)

        return loss

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(
            state_spec,
            grad_batch_spec
            if training_args.gradient_accumulation_steps > 1
            else batch_spec,
            None,
        ),
        out_axis_resources=(state_spec, None),
        donate_argnums=(0,),
    )
    p_eval_step = pjit(
        eval_step,
        in_axis_resources=(state_spec, batch_spec),
        out_axis_resources=None,
    )

    # init variables
    last_time = time.perf_counter()
    train_metrics = None
    step = int(state.step)
    metrics_logger = MetricsLogger(step)
    epochs = tqdm(
        range(state.epoch, num_epochs),
        desc=f"Epoch ... (1/{num_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )

    def run_evaluation():
        # ======================== Evaluating ==============================
        if training_args.do_eval:
            eval_loader = dataset.dataloader("eval", eval_batch_size_per_step)
            eval_steps = (
                len_eval_dataset // eval_batch_size_per_step
                if len_eval_dataset is not None
                else None
            )
            eval_loss = []
            for batch in tqdm(
                eval_loader,
                desc="Evaluating...",
                position=2,
                leave=False,
                total=eval_steps,
                disable=jax.process_index() > 0,
            ):
                # need to keep only eval_batch_size_per_node items relevant to the node
                batch = jax.tree_map(
                    lambda x: x.reshape(
                        (jax.process_count(), eval_batch_size_per_node) + x.shape[1:]
                    ),
                    batch,
                )
                batch = jax.tree_map(lambda x: x[jax.process_index()], batch)

                # add dp dimension when using "vmap trick"
                if use_vmap_trick:
                    bs_shape = (
                        jax.local_device_count() // training_args.mp_devices,
                        training_args.per_device_eval_batch_size,
                    )
                    batch = jax.tree_map(
                        lambda x: x.reshape(bs_shape + x.shape[1:]), batch
                    )

                # freeze batch to pass safely to jax transforms
                batch = freeze(batch)
                # accumulate losses async
                eval_loss.append(p_eval_step(state, batch))

            # get the mean of the loss
            eval_loss = jnp.stack(eval_loss)
            eval_loss = jnp.mean(eval_loss)
            eval_metrics = {"loss": eval_loss}

            # log metrics
            metrics_logger.log(eval_metrics, prefix="eval")

            # Print metrics and update progress bar
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

            return eval_metrics

    def run_save_model(state, eval_metrics=None):
        if jax.process_index() == 0:

            output_dir = training_args.output_dir
            use_bucket = output_dir.startswith("gs://")
            if use_bucket:
                bucket_path = Path(output_dir[5:]) / wandb.run.id / f"step_{state.step}"
                bucket, dir_path = str(bucket_path).split("/", 1)
                tmp_dir = tempfile.TemporaryDirectory()
                output_dir = tmp_dir.name

            # save model
            params = jax.device_get(state.params)
            model.save_pretrained(
                output_dir,
                params=params,
            )

            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # copy to bucket
            if use_bucket:
                client = storage.Client()
                bucket = client.bucket(bucket)
                for filename in Path(output_dir).glob("*"):
                    blob_name = str(Path(dir_path) / "model" / filename.name)
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(filename))
                tmp_dir.cleanup()

            # save state
            opt_state = jax.device_get(state.opt_state)
            if use_bucket:
                blob_name = str(Path(dir_path) / "state" / "opt_state.msgpack")
                blob = bucket.blob(blob_name)
                blob.upload_from_file(io.BytesIO(to_bytes(opt_state)))
            else:
                with (Path(output_dir) / "opt_state.msgpack").open("wb") as f:
                    f.write(to_bytes(opt_state))

            # save to W&B
            if training_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("20GB"))

                metadata = {
                    k: jax.device_get(getattr(state, k)).item()
                    for k in ["step", "epoch", "train_time", "train_samples"]
                }
                metadata["num_params"] = num_params
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics

                # create model artifact
                if use_bucket:
                    metadata["bucket_path"] = f"gs://{bucket_path}/model"
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="DalleBart_model",
                    metadata=metadata,
                )
                if use_bucket:
                    artifact.add_reference(metadata["bucket_path"])
                else:
                    for filename in [
                        "config.json",
                        "flax_model.msgpack",
                        "merges.txt",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    ]:
                        artifact.add_file(
                            f"{Path(training_args.output_dir) / filename}"
                        )
                wandb.run.log_artifact(artifact)

                # create state artifact
                if use_bucket:
                    metadata["bucket_path"] = f"gs://{bucket_path}/state"
                artifact_state = wandb.Artifact(
                    name=f"state-{wandb.run.id}",
                    type="DalleBart_state",
                    metadata=metadata,
                )
                if use_bucket:
                    artifact_state.add_reference(metadata["bucket_path"])
                else:
                    artifact_state.add_file(
                        f"{Path(training_args.output_dir) / 'opt_state.msgpack'}"
                    )
                wandb.run.log_artifact(artifact_state)

    logger.info("  Ready to start training")
    with maps.mesh(mesh.devices, mesh.axis_names):
        for epoch in epochs:
            state.replace(epoch=epoch)
            # ======================== Training ================================
            metrics_logger.update_state_metrics(state)
            metrics_logger.log({})

            # Generate an epoch by shuffling sampling indices from the train dataset
            train_loader = dataset.dataloader(
                "train",
                batch_size_per_node,
                epoch,
            )
            # train
            for batch in tqdm(
                train_loader,
                desc="Training...",
                position=1,
                leave=False,
                total=steps_per_epoch,
                disable=jax.process_index() > 0,
            ):
                # calculate delta time (we have a lag of one step but it's ok)
                new_time = time.perf_counter()
                delta_time = new_time - last_time
                last_time = new_time

                # set correct shape to batch
                # - add grad_step dim if gradient_accumulation_steps > 1
                # - split per dp device if not multi-host for vmap trick (does not work in multi-host)
                bs_shape = (
                    (batch_size_per_node_per_grad_step,)
                    if not use_vmap_trick
                    else (
                        jax.local_device_count()
                        // training_args.mp_devices,  # local dp devices
                        training_args.per_device_train_batch_size,
                    )
                )
                if training_args.gradient_accumulation_steps > 1:
                    # reshape data into (gradient_accumulation_steps, batch_per_node, ...)
                    # to avoid any data redistribution when sharding
                    bs_shape = (training_args.gradient_accumulation_steps,) + bs_shape

                # reshape batch
                batch = jax.tree_map(
                    lambda x: x.reshape(bs_shape + x.shape[1:]),
                    batch,
                )
                # freeze batch to pass safely to jax transforms
                batch = freeze(batch)

                # train step
                state, train_metrics = p_train_step(state, batch, delta_time)
                step += 1

                if step % training_args.logging_steps == 0 and jax.process_index() == 0:
                    metrics_logger.update_state_metrics(state)
                    metrics_logger.log(train_metrics, prefix="train")

                eval_metrics = None
                if step % training_args.eval_steps == 0:
                    eval_metrics = run_evaluation()

                if step % training_args.save_steps == 0:
                    run_save_model(state, eval_metrics)

            # log final train metrics
            if train_metrics is not None:
                metrics_logger.update_state_metrics(state)
                metrics_logger.log(train_metrics, prefix="train")

                epochs.write(
                    f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metrics['loss']}, Learning Rate: {train_metrics['learning_rate']})"
                )

            # Final evaluation
            eval_metrics = run_evaluation()

            # save checkpoint after each epoch
            run_save_model(state, eval_metrics)


if __name__ == "__main__":
    main()
