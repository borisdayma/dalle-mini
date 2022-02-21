#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-2022 The HuggingFace & DALL·E Mini Team All rights reserved.
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

import json
import logging
import os
import sys
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
from distributed_shampoo import GraftingType, distributed_shampoo
from flax.core.frozen_dict import FrozenDict, freeze
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import onehot
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from tqdm import tqdm
from transformers import HfArgumentParser

from dalle_mini.data import Dataset
from dalle_mini.model import (
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer,
    set_partitions,
)

cc.initialize_cache(
    "/home/boris/dalle-mini/jax_cache", max_cache_size_bytes=5 * 2 ** 30
)


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
            "help": "Restore optimizer and training state associated with a wandb checkpoint."
        },
    )

    state_artifact: str = field(init=False)

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name == self.model_name_or_path
            assert (
                self.tokenizer_name is not None
            ), "Tokenizer name or model name/path needs to be specified"
        if self.restore_state:
            assert self.model_name_or_path is not None and (
                "/model-" in self.model_name_or_path
            ), "Restoring state only available with W&B artifact reference"
            self.state_artifact = self.model_name_or_path.replace(
                "/model-", "/state-", 1
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
    start_preconditioning_step: int = field(
        default=100,
        metadata={"help": "Number of steps before starting to update preconditioner."},
    )
    preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
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
        assert self.optim in [
            "distributed_shampoo",
            "adam",
            "adafactor",
        ], f"Selected optimizer not supported: {self.optim}"
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
            log_metrics = {
                f"{prefix}/{k}" if prefix is not None else k: v
                for k, v in metrics.items()
            }
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
    if training_args.assert_TPU_available:
        assert (
            jax.local_device_count() == 8
        ), "TPUs in use, please check running processes"

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
    else:
        config = None

    # Load or create new model
    if model_args.model_name_or_path:
        model = DalleBart.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            abstract_init=True,
            load_on_cpu=True,
        )
    else:
        model = DalleBart(
            config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            load_on_cpu=True,
        )

    # update model config per training args
    model.config.gradient_checkpointing = training_args.gradient_checkpointing

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
        len_train_dataset // batch_size_per_step
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
        f"  Batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(
        f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Batch size per update = {batch_size_per_step}")
    logger.info(f"  Model parameters = {num_params:,}")

    # create wandb run
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
            schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
        )
        return schedule_fn

    learning_rate_fn = create_learning_rate_fn()

    # create adam optimizer
    if training_args.optim == "distributed_shampoo":
        # parameters from https://github.com/tensorflow/lingvo/blob/03ee9d7cd50764b0424c7c863733c91fc0b053ec/lingvo/jax/optimizers.py#L729
        optimizer = distributed_shampoo(
            learning_rate_fn,
            block_size=training_args.block_size,
            beta1=training_args.beta1,
            beta2=training_args.beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=1e-8,
            start_preconditioning_step=training_args.start_preconditioning_step,
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=GraftingType.RMSPROP_NORMALIZED,
            nesterov=False,
            exponent_override=0,
            statistics_partition_spec=PartitionSpec(None, "batch", None),
            preconditioner_partition_spec=PartitionSpec("batch", None, None),
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
                partition_spec_for_statistics=PartitionSpec(None, "batch", None),
            )
        else:
            raise NotImplementedError
        return opt_state_spec, opt_state_shape

    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape(param_spec)

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("batch", "mp"))

    # define state spec
    state_spec = TrainState(
        params=param_spec,
        opt_state=opt_state_spec,
        dropout_rng=None,
        step=None,
        epoch=None,
        train_time=None,
        train_samples=None,
        apply_fn=model.__call__,
        tx=optimizer,
    )

    # create training state
    with maps.mesh(mesh.devices, mesh.axis_names):
        if not model_args.restore_state:

            def init_state(params):
                return TrainState.create(
                    apply_fn=model.__call__,
                    tx=optimizer,
                    params=params,
                    dropout_rng=dropout_rng,
                )

            state = pjit(
                init_state,
                in_axis_resources=(param_spec,),
                out_axis_resources=state_spec,
                donate_argnums=(0,),
            )(model.params)

        else:
            # get state files from artifact
            if jax.process_index() == 0:
                artifact = wandb.run.use_artifact(model_args.state_artifact)
            else:
                artifact = wandb.Api().artifact(model_args.state_artifact)
            artifact_dir = artifact.download()

            # restore opt_state
            with (Path(artifact_dir) / "opt_state.msgpack").open("rb") as f:
                opt_state = from_bytes(opt_state_shape, f.read())

            # restore other attributes
            with (Path(artifact_dir) / "training_state.json").open("r") as f:
                attr_state = json.load(f)

            def restore_state(params, opt_state):
                return TrainState(
                    apply_fn=model.__call__,
                    tx=optimizer,
                    params=params,
                    opt_state=opt_state,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                restore_state,
                in_axis_resources=(param_spec, opt_state_spec),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(model.params, opt_state)

            # remove opt_state from CPU
            del opt_state

    # free memory
    del model._params, opt_state_spec, opt_state_shape

    # define batch specs
    keys = ["attention_mask", "decoder_input_ids", "input_ids", "labels"]
    batch_spec = freeze({k: PartitionSpec("batch") for k in keys})
    grad_batch_spec = freeze({k: PartitionSpec(None, "batch") for k in keys})

    # label smoothed cross entropy
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # Define gradient update step fn
    def train_step(state, batch, delta_time):
        # we reshape to (gradient_accumulation_steps, dp_devices, ...)
        # allows feeding partial batch size per node for full model parallel
        batch = jax.tree_map(
            lambda x: x.reshape(
                (
                    training_args.gradient_accumulation_steps,
                    training_args.dp_devices,
                    training_args.per_device_train_batch_size,
                )
                + x.shape[2:]
            ),
            batch,
        )
        # ensure data is sharded correctly per dp device
        batch = with_sharding_constraint(batch, grad_batch_spec)

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
            # minibatch at grad_idx, shape (dp_devices, per_device_train_batch_size, ...)
            minibatch = get_minibatch(batch, grad_idx)
            # calculate loss and grads independently per dp_device
            dropout_rng, _ = jax.random.split(dropout_rng)
            # ensure inputs are sharded per device
            minibatch = jax.tree_map(
                lambda x: with_sharding_constraint(x, PartitionSpec("batch")),
                minibatch,
            )
            # only 1 single rng per grad step, let us handle larger batch size
            loss_grads = jax.vmap(grad_fn, in_axes=(None, 0, None), out_axes=(0, 0))(
                state.params, minibatch, dropout_rng
            )
            # ensure outputs are sharded per device
            loss_grads = jax.tree_map(
                lambda x: with_sharding_constraint(x, PartitionSpec("batch")),
                loss_grads,
            )
            # average across all devices
            loss_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_grads)
            # return loss and grads
            return loss_grads, dropout_rng

        if training_args.gradient_accumulation_steps == 1:
            loss_grad, dropout_rng = loss_and_grad(0, state.dropout_rng)
        else:
            # create initial state for cumul_minibatch_step loop
            init_minibatch_step = (
                (
                    0.0,
                    jax.tree_map(jnp.zeros_like, state.params),
                ),
                state.dropout_rng,
            )

            # accumulate gradients
            def cumul_minibatch_step(grad_idx, cumul_loss_grad_dropout):
                cumul_loss_grad, dropout_rng = cumul_loss_grad_dropout
                loss_grad, dropout_rng = loss_and_grad(grad_idx, dropout_rng)
                cumul_loss_grad = jax.tree_map(jnp.add, cumul_loss_grad, loss_grad)
                return cumul_loss_grad, dropout_rng

            # loop over gradients
            loss_grad, dropout_rng = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_minibatch_step,
            )
            # sum -> mean
            loss_grad = jax.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps, loss_grad
            )

        # update state
        loss, grads = loss_grad
        state = state.apply_gradients(
            grads=grads,
            dropout_rng=dropout_rng,
            train_time=state.train_time + delta_time,
            train_samples=state.train_samples + batch_size_per_step,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }

        return state, metrics

    # Define eval fn
    def eval_step(state, batch):
        # we reshape to (dp_devices, ...)
        batch = jax.tree_map(
            lambda x: x.reshape(
                (
                    training_args.dp_devices,
                    training_args.per_device_eval_batch_size,
                )
                + x.shape[1:]
            ),
            batch,
        )
        # ensure data is sharded correctly per dp device
        batch = with_sharding_constraint(batch, batch_spec)

        def compute_eval_loss(batch):
            batch, labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=state.params, train=False)[0]
            return loss_fn(logits, labels)

        # calculate loss independently per dp_device
        loss = jax.vmap(compute_eval_loss, in_axes=(0,), out_axes=0)(batch)
        # ensure they are sharded over dp devices
        loss = with_sharding_constraint(loss, PartitionSpec("batch"))
        # average across all devices
        loss = jnp.mean(loss)
        return loss

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(state_spec, grad_batch_spec, None),
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
        range(state.epoch, num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0
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
            ):
                # need to keep only eval_batch_size_per_node items relevant to the node
                batch = jax.tree_map(
                    lambda x: x.reshape(
                        (jax.process_count(), eval_batch_size_per_node) + x.shape[1:]
                    ),
                    batch,
                )
                batch = jax.tree_map(lambda x: x[jax.process_index()], batch)
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
            params = jax.device_get(state.params)
            # save model locally
            model.save_pretrained(
                training_args.output_dir,
                params=params,
            )

            # save tokenizer
            tokenizer.save_pretrained(training_args.output_dir)

            # save state
            opt_state = jax.device_get(state.opt_state)
            with (Path(training_args.output_dir) / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(opt_state))
            state_dict = {
                k: jax.device_get(getattr(state, k)).item()
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
                metadata["num_params"] = num_params
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics

                # create model artifact
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="DalleBart_model",
                    metadata=metadata,
                )
                for filename in [
                    "config.json",
                    "flax_model.msgpack",
                    "merges.txt",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ]:
                    artifact.add_file(f"{Path(training_args.output_dir) / filename}")
                wandb.run.log_artifact(artifact)

                # create state artifact
                artifact_state = wandb.Artifact(
                    name=f"state-{wandb.run.id}",
                    type="DalleBart_state",
                    metadata=metadata,
                )
                for filename in ["opt_state.msgpack", "training_state.json"]:
                    artifact_state.add_file(
                        f"{Path(training_args.output_dir) / filename}"
                    )
                wandb.run.log_artifact(artifact_state)

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
            ):
                # calculate delta time (we have a lag of one step but it's ok)
                new_time = time.perf_counter()
                delta_time = new_time - last_time
                last_time = new_time

                # reshape data into (gradient_accumulation_steps, dp_devices, batch_per_dp, ...)
                batch = jax.tree_map(
                    lambda x: x.reshape(
                        (
                            training_args.gradient_accumulation_steps,
                            batch_size_per_node_per_grad_step,
                        )
                        + x.shape[1:]
                    ),
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
