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
# set a common huggingface cache folder (used with datasets and transformers) and wandb cache folder (used with artifacts)
os.environ['HF_HOME'] = '/data/huggingface/'     # required before importing transformers & datasets
os.environ['WANDB_CACHE_DIR'] = '/data/wandb/'   # required before importing wandb

import logging as pylogging    # To avoid collision with transformers.utils.logging
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from filelock import FileLock
from flax import jax_utils, traverse_util
import flax.linen as nn
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    FlaxBartForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.models.bart.modeling_flax_bart import *
from transformers.file_utils import is_offline_mode

import wandb

logger = pylogging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Model hyperparameters, for convenience
OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'facebook/bart-large-cnn'


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
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='caption',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    encoding_column: Optional[str] = field(
        default='encoding',
        metadata={"help": "The name of the column in the datasets containing the image encodings."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input predict data file to do prediction on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    no_decay: bool = field(
        default=False, metadata={"help": "Whether to use decay in the learning rate scheduler."}
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=80,     # ensure we have the same datasets cached data and avoid using too much space
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
            "which is used during evaluation."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    log_interval: Optional[int] = field(
        default=40,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    log_model: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["tsv", "csv", "json"], "`train_file` should be a tsv, csv or json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["tsv", "csv", "json"], "`validation_file` should be a tsv, csv or json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray
    grad_accum: jnp.ndarray
    optimizer_step: int

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # we keep shared to easily load pre-trained weights
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        # a separate embedding is used for the decoder
        self.decoder_embed = nn.Embed(
            OUTPUT_VOCAB_SIZE,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

        # the decoder has a different config
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = OUTPUT_LENGTH
        decoder_config.min_length = OUTPUT_LENGTH
        decoder_config.max_length = OUTPUT_LENGTH
        decoder_config.vocab_size = OUTPUT_VOCAB_SIZE
        self.decoder = FlaxBartDecoder(decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed)

class CustomFlaxBartForConditionalGenerationModule(FlaxBartForConditionalGenerationModule):
    def setup(self):
        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            OUTPUT_VOCAB_SIZE,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, OUTPUT_VOCAB_SIZE))

class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule
    

def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False):
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


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float, no_decay: bool
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    if no_decay:
        return warmup_fn
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def wandb_log(metrics, step=None, prefix=None):
    if jax.process_index() == 0:
        log_metrics = {f'{prefix}/k' if prefix is not None else k: jax.device_get(v) for k,v in metrics.items()}
        if step is not None:
            log_metrics = {**metrics, 'train/step': step}
        wandb.log(log_metrics)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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
    
    # Set up wandb run
    wandb.init(
        entity='wandb',
        project='hf-flax-dalle-mini',
        job_type='Seq2SeqVQGAN',
        config=parser.parse_args()
    )

    # set default x-axis as 'train/step'
    wandb.define_metric('train/step')
    wandb.define_metric('*', step_metric='train/step')

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
    data_files = {}
    logger.warning(f"Datasets path have been manually hardcoded")  # TODO: remove it later, convenient for now
    if data_args.train_file is not None:
        data_files["train"] = ["/data/CC3M/training-encoded.tsv", "/data/CC12M/encoded-train.tsv"]
    if data_args.validation_file is not None:
        data_files["validation"] = ["/data/CC3M/validation-encoded.tsv"]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    dataset = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir, delimiter="\t")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    base_model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    )

    # Set up our new model config
    config = BartConfig.from_pretrained(model_args.model_name_or_path)
    config.tie_word_embeddings = False
    config.decoder_start_token_id = BOS_TOKEN_ID
    config.bos_token_id = BOS_TOKEN_ID  # should not be used
    config.pos_token_id = BOS_TOKEN_ID  # should not be needed (as we generate until max_length)
    config.eos_token_id = BOS_TOKEN_ID + 1  # unreachable
    config.forced_bos_token_id = None  # we don't need this token
    config.forced_eos_token_id = None  # we don't need this token
    #config.min_length = data_args.max_target_length        # Set only in decoder?
    #config.max_length = data_args.max_target_length        # Set only in decoder?

    print(f"TPUs: {jax.device_count()}")
    assert jax.device_count() == 8, "TPUs in use, please check running processes"

    # Create a custom model and initialize it randomly
    model = CustomFlaxBartForConditionalGeneration(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype))

    # Use pre-trained weights for encoder
    model.params['model']['encoder'] = base_model.params['model']['encoder']
    model.params['model']['shared'] = base_model.params['model']['shared']
    del base_model

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    text_column = data_args.text_column
    encoding_column = data_args.encoding_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def shift_tokens_right(input_ids: np.array, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id
        return shifted_input_ids

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
	# Setting padding="max_length" as we need fixed length inputs for jitted functions
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True, return_tensors="np"
        )

        # set up targets
        # Note: labels correspond to our target indices
        # decoder input ids are the same but shifted to the right with bos at the beginning (and without last token)
        labels = [eval(indices) for indices in examples['encoding']]
        labels = np.asarray(labels)

        # We need the labels, in addition to the decoder_input_ids, for the compute_loss function
        model_inputs["labels"] = labels

        # In our case, this prepends the bos token and removes the last one
        decoder_input_ids = shift_tokens_right(labels, config.decoder_start_token_id)
        model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Metric
    #metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    total_batch_size = int(train_batch_size) * training_args.gradient_accumulation_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_steps = steps_per_epoch * num_epochs
    total_optimization_steps = (len(train_dataset) // total_batch_size) * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        total_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
        data_args.no_decay
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
            (name, "scale") for name in ["self_attn_layer_norm", "layernorm_embedding", "final_layer_norm"]
        ]
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # Setup train state
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
        grad_accum=jax.tree_map(jnp.zeros_like, model.params),
        optimizer_step=0,
    )

    # label smoothed cross entropy
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params)
        grad_accum = jax.tree_multimap(lambda x, y: x + y, grads, state.grad_accum)

        def update_fn():
            grads = jax.tree_map(lambda x: x / training_args.gradient_accumulation_steps, grad_accum)
            grads = jax.lax.pmean(grads, "batch")
            new_state = state.apply_gradients(
                grads=grads, grad_accum=jax.tree_map(jnp.zeros_like, grads), optimizer_step=state.optimizer_step + 1
            )
            return new_state

        new_state = jax.lax.cond(
            (state.step + 1) % training_args.gradient_accumulation_steps == 0,
            lambda _: update_fn(),
            lambda _: state.replace(grad_accum=grad_accum, step=state.step + 1),
            None,
        )

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.optimizer_step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state.replace(dropout_rng=new_dropout_rng), metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Define generation function
    max_length = (
        data_args.val_max_target_length if data_args.val_max_target_length is not None else model.config.max_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
        return output_ids.sequences

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(
        train_step, "batch", donate_argnums=(0,)
    )
    p_eval_step = jax.pmap(eval_step, "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total global steps = {total_steps}")
    logger.info(f"  Total optimization steps = {total_optimization_steps}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    global_step = 0

    def run_evaluation():
        # ======================== Evaluating ==============================
        eval_metrics = []
        if training_args.do_eval:
            eval_preds = []
            eval_labels = []

            eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size)
            eval_steps = len(eval_dataset) // eval_batch_size
            for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                # Model forward
                batch = next(eval_loader)
                labels = batch["labels"]

                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

                # generation
                if data_args.predict_with_generate:
                    generated_ids = p_generate_step(state.params, batch)
                    eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                    eval_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # log metrics
            wandb_log(eval_metrics, step=global_step, prefix='eval')

            # compute ROUGE metrics
            rouge_desc = ""
        #    if data_args.predict_with_generate:
        #        rouge_metrics = compute_metrics(eval_preds, eval_labels)
        #        eval_metrics.update(rouge_metrics)
        #        rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])

            # Print metrics and update progress bar
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']} | {rouge_desc})"
            epochs.write(desc)
            epochs.desc = desc

            return eval_metrics

    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
        steps_per_epoch = len(train_dataset) // train_batch_size
        # train
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            global_step +=1
            batch = next(train_loader)
            state, train_metric = p_train_step(state, batch)

            if global_step % data_args.log_interval == 0 and jax.process_index() == 0:
                # log metrics
                wandb_log(unreplicate(train_metric), step=global_step, prefix='tran')

            if global_step % training_args.eval_steps == 0:
                run_evaluation()

        train_time += time.time() - train_start
        train_metric = unreplicate(train_metric)
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
        )

        # Final evaluation
        eval_metrics = run_evaluation()

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))

            # save model locally
            model.save_pretrained(
                training_args.output_dir,
                params=params,
            )

            # save to W&B
            if data_args.log_model:
                metadata = {'epoch': epoch+1, 'eval/loss': eval_metrics['loss']}
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", type="bart_model", metadata=metadata
                )
                artifact.add_file(str(Path(training_args.output_dir) / 'flax_model.msgpack'))
                artifact.add_file(str(Path(training_args.output_dir) / 'config.json'))
                wandb.run.log_artifact(artifact)

            # save to the hub
            if training_args.push_to_hub:
                model.save_pretrained(
                    training_args.output_dir,
                    params=params,
                    push_to_hub=training_args.push_to_hub,
                    commit_message=f"Saving weights and logs of epoch {epoch+1}",
                    temp_dir=True  # avoid issues with being in a repository
                )

    # ======================== Prediction loop ==============================
    if training_args.do_predict:
        logger.info("*** Predict ***")

        pred_metrics = []
        pred_generations = []
        pred_labels = []

        pred_loader = data_loader(input_rng, predict_dataset, eval_batch_size)
        pred_steps = len(predict_dataset) // eval_batch_size
        for _ in tqdm(range(pred_steps), desc="Predicting...", position=2, leave=False):
            # Model forward
            batch = next(pred_loader)
            labels = batch["labels"]

            metrics = p_eval_step(state.params, batch)
            pred_metrics.append(metrics)

            # generation
            if data_args.predict_with_generate:
                generated_ids = p_generate_step(state.params, batch)
                pred_generations.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                pred_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))

        # normalize prediction metrics
        pred_metrics = get_metrics(pred_metrics)
        pred_metrics = jax.tree_map(jnp.mean, pred_metrics)

        # compute ROUGE metrics
        rouge_desc = ""
        if data_args.predict_with_generate:
            rouge_metrics = compute_metrics(pred_generations, pred_labels)
            pred_metrics.update(rouge_metrics)
            rouge_desc = " ".join([f"Predict {key}: {value} |" for key, value in rouge_metrics.items()])

        # Print metrics
        desc = f"Predict Loss: {pred_metrics['loss']} | {rouge_desc})"
        logger.info(desc)


if __name__ == "__main__":
    main()
