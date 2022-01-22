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

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from datasets import Dataset
from distributed_shampoo import GraftingType, distributed_shampoo
from flax.core.frozen_dict import freeze
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import onehot, stack_forest
from jax.experimental import PartitionSpec, maps
from jax.experimental.pjit import pjit
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

import wandb
from dalle_mini.data import Dataset
from dalle_mini.model import (
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer,
    set_partitions,
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
        default=8, metadata={"help": "Batch size per GPU/TPU/CPU for training."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size per GPU/TPU/CPU for evaluation. Same as training batch size if not set."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing an update pass."
        },
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

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Reference to a wandb artifact for resuming training."},
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


class MetricsLogger:
    def __init__(self, state):
        self.step = state.step
        self.time = time.perf_counter()

    def get_all_train_metrics(self, train_metrics, state):
        """Make a dict of training metrics to be logged"""
        metrics = train_metrics
        # get state parameters
        state_dict = {
            k.split("_")[-1]: getattr(state, k)
            for k in ["epoch", "train_time", "train_samples"]
        }
        # timing metrics
        new_step = int(state.step)
        new_time = time.perf_counter()
        if new_step > self.step:
            time_per_step = (new_time - self.time) / (new_step - self.step)
            self.step = new_step
            self.time = new_time
            state_dict["time_per_step"] = time_per_step
        return {**metrics, **state_dict}

    @staticmethod
    def log(metrics, step=None, prefix=None):
        if jax.process_index() == 0:
            log_metrics = {
                f"{prefix}/{k}" if prefix is not None else k: v
                for k, v in metrics.items()
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

    if training_args.resume_from_checkpoint is not None:
        if jax.process_index() == 0:
            artifact = wandb.run.use_artifact(training_args.resume_from_checkpoint)
        else:
            artifact = wandb.Api().artifact(training_args.resume_from_checkpoint)
        artifact_dir = artifact.download()

        # load model
        model = DalleBart.from_pretrained(
            artifact_dir,
            dtype=getattr(jnp, model_args.dtype),
            abstract_init=True,
        )

        # load tokenizer
        tokenizer = DalleBartTokenizer.from_pretrained(
            artifact_dir,
            use_fast=True,
        )

    else:
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
            )
        else:
            model = DalleBart(
                config,
                seed=training_args.seed_model,
                dtype=getattr(jnp, model_args.dtype),
            )

        # Load tokenizer
        if model_args.tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, use_fast=True
            )
        else:
            tokenizer = DalleBartTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
            )

    # Preprocessing the datasets.
    # We need to normalize and tokenize inputs and targets.

    dataset.preprocess(
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        normalize_text=model.config.normalize_text,
        max_length=model.config.max_text_length,
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed_model)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = training_args.num_train_epochs
    # batch size per node
    train_batch_size = (
        training_args.per_device_train_batch_size * jax.local_device_count()
    )
    batch_size_per_node = train_batch_size * training_args.gradient_accumulation_steps
    batch_size_per_step = batch_size_per_node * jax.process_count()
    eval_batch_size = (
        training_args.per_device_eval_batch_size * jax.local_device_count()
    )
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
            start_preconditioning_step=training_args.warmup_steps,
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=GraftingType.RMSPROP_NORMALIZED,
            nesterov=False,
            exponent_override=0,
            batch_axis_name="batch",
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=True,
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=None,
            precision=jax.lax.Precision.HIGHEST,
            best_effort_memory_usage_reduction=training_args.optim_quantized,
        )

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

    # get opt_state shape without actual init
    param_shape = jax.tree_map(lambda x: x.shape, model.params)
    opt_state_shape = jax.eval_shape(lambda x: optimizer.init(x), param_shape)

    # get PartitionSpec for model params
    param_spec = set_partitions(model.params)

    # create PartitionSpec for opt_state
    def opt_state_spec_per_leaf(x):
        if training_args.optim in ["adam", "adafactor"]:
            if isinstance(x, dict):
                # variables with same structure as params
                return param_spec
            else:
                # other variables such as count
                return None
        else:
            # TODO: create spec for Distributed Shampoo
            raise NotImplementedError

    opt_state_spec = jax.tree_map(
        opt_state_spec_per_leaf,
        opt_state_shape,
        # return None spec for empty elements
        is_leaf=lambda x: isinstance(x, (dict, optax.EmptyState)),
    )

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("batch", "mp"))

    # Create state spec
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
    def init_state(params):
        state = TrainState.create(
            apply_fn=model.__call__,
            tx=optimizer,
            params=freeze(params),
            dropout_rng=dropout_rng,
        )
        return state

    with maps.mesh(mesh.devices, mesh.axis_names):
        state = pjit(
            init_state,
            in_axis_resources=None,
            out_axis_resources=state_spec,
            donate_argnums=(0,),
        )(freeze(model.params))

    if training_args.resume_from_checkpoint is not None:
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
        # use a different rng per node
        dropout_rng = jax.random.fold_in(dropout_rng, jax.process_index())

        def compute_loss(params, minibatch):
            labels = minibatch.pop("labels")
            logits = state.apply_fn(
                **minibatch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            return loss_fn(logits, labels)

        grad_fn = jax.value_and_grad(compute_loss)

        if training_args.gradient_accumulation_steps == 1:
            minibatch = jax.tree_map(lambda x: x[0], batch)
            loss, grads = grad_fn(state.params, minibatch)
        else:

            def _cumul_loss_grads(i, cumul_loss_grads):
                minibatch = jax.tree_map(lambda x: x[i], batch)
                return jax.tree_map(
                    lambda x, y: x + y,
                    cumul_loss_grads,
                    grad_fn(state.params, minibatch),
                )

            init_loss_grads = (
                0.0,
                jax.tree_map(jnp.zeros_like, state.params),
            )
            loss, grads = jax.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps,
                jax.lax.fori_loop(
                    0,
                    training_args.gradient_accumulation_steps,
                    _cumul_loss_grads,
                    init_loss_grads,
                ),
            )

        state = state.apply_gradients(
            grads=grads,
            dropout_rng=new_dropout_rng,
            train_time=state.train_time + delta_time,
            train_samples=state.train_samples + batch_size_per_step,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }

        return state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        metrics = {"loss": loss}
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(state_spec, PartitionSpec("batch", None), None),
        out_axis_resources=(state_spec, None),
        donate_argnums=(0,),
    )
    p_eval_step = pjit(
        eval_step,
        in_axis_resources=(param_spec, PartitionSpec("batch", None)),
        out_axis_resources=None,
    )

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
    epochs = tqdm(
        range(state.epoch, num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0
    )

    metrics_logger = MetricsLogger(state)
    if jax.process_index() == 0:
        # set default x-axis as 'train/step'
        metrics_logger.log({}, step=state.step)
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

    def run_evaluation():
        # ======================== Evaluating ==============================
        eval_metrics = []
        if training_args.do_eval:
            eval_loader = dataset.dataloader(
                "eval", training_args.per_device_eval_batch_size
            )
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
            eval_metrics = stack_forest(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # log metrics
            metrics_logger.log(eval_metrics, step=state.step, prefix="eval")

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

            if jax.process_index() == 0:
                # save to W&B
                if training_args.log_model:
                    # save some space
                    c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                    c.cleanup(wandb.util.from_human_size("10GB"))

                    metadata = dict(state_dict)
                    metadata["num_params"] = num_params
                    if eval_metrics is not None:
                        metadata["eval"] = eval_metrics
                    artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="bart_model",
                        metadata=metadata,
                    )
                    artifact.add_file(
                        str(Path(training_args.output_dir) / "flax_model.msgpack")
                    )
                    artifact.add_file(
                        str(Path(training_args.output_dir) / "config.json")
                    )
                    artifact.add_file(
                        str(Path(training_args.output_dir) / "tokenizer.json")
                    )
                    artifact.add_file(
                        str(Path(training_args.output_dir) / "tokenizer_config.json")
                    )
                    artifact.add_file(
                        str(Path(training_args.output_dir) / "vocab.json")
                    )
                    artifact.add_file(
                        str(Path(training_args.output_dir) / "merges.txt")
                    )
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

    # init variables
    last_time = time.perf_counter()
    train_metrics = None

    with maps.mesh(mesh.devices, mesh.axis_names):
        for epoch in epochs:
            state.replace(epoch=epoch)
            # ======================== Training ================================
            metrics_logger.log({"train/epoch": epoch}, step=state.step)

            # Generate an epoch by shuffling sampling indices from the train dataset
            train_loader = dataset.dataloader(
                "train",
                training_args.per_device_train_batch_size,
                training_args.gradient_accumulation_steps,
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

                # train step
                state, train_metrics = p_train_step(state, batch, delta_time)
                step = state.step

                if step % training_args.logging_steps == 0 and jax.process_index() == 0:
                    all_metrics = metrics_logger.get_all_train_metrics(
                        train_metrics, state
                    )
                    metrics_logger.log(all_metrics, step=step, prefix="train")

                eval_metrics = None
                if training_args.eval_steps and step % training_args.eval_steps == 0:
                    eval_metrics = run_evaluation()

                if step % training_args.save_steps == 0:
                    run_save_model(state, eval_metrics)

            # log final train metrics
            if train_metrics is not None:
                all_metrics = metrics_logger.get_all_train_metrics(train_metrics, state)
                metrics_logger.log(all_metrics, step=step, prefix="train")

                epochs.write(
                    f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metrics['loss']}, Learning Rate: {train_metrics['learning_rate']})"
                )

            # Final evaluation
            eval_metrics = run_evaluation()

            # save checkpoint after each epoch
            run_save_model(state, eval_metrics)


if __name__ == "__main__":
    main()
