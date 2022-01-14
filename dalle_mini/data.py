from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from braceexpand import braceexpand
from datasets import Dataset, load_dataset
from flax.training.common_utils import shard

from .text import TextNormalizer


@dataclass
class Dataset:
    dataset_repo_or_path: str
    train_file: str = None
    validation_file: str = None
    streaming: bool = True
    use_auth_token: bool = False
    text_column: str = "caption"
    encoding_column: str = "encoding"
    max_train_samples: int = None
    max_eval_samples: int = None
    preprocessing_num_workers: int = None
    overwrite_cache: bool = False
    do_train: bool = False
    do_eval: bool = True
    seed_dataset: int = None
    shard_by_host: bool = False
    train_dataset: Dataset = field(init=False)
    eval_dataset: Dataset = field(init=False)
    rng_dataset: jnp.ndarray = field(init=False)
    multi_hosts: bool = field(init=False)

    def __post_init__(self):
        self.multi_hosts = jax.process_count() > 1
        # define data_files
        if self.train_file is not None or self.validation_file is not None:
            # accept braceexpand notation
            for k in ["train_file", "validation_file"]:
                f = getattr(self, k)
                if isinstance(f, str):
                    setattr(self, k, list(braceexpand(f)))
            # for list of files, split training data shards by host
            if (
                isinstance(self.train_file, list)
                and self.multi_hosts
                and self.shard_by_host
            ):
                self.train_file = self.train_file[
                    jax.process_index() :: jax.process_count()
                ]
            data_files = {
                "train": self.train_file,
                "validation": self.validation_file,
            }
        else:
            data_files = None

        # load dataset
        dataset = load_dataset(
            self.dataset_repo_or_path,
            data_files=data_files,
            streaming=self.streaming,
            use_auth_token=self.use_auth_token,
        )
        if self.do_train:
            if "train" not in dataset:
                raise ValueError("Training requires a training dataset")
            self.train_dataset = dataset["train"]
            if self.max_train_samples is not None:
                self.train_dataset = (
                    self.train_dataset.take(self.max_train_samples)
                    if self.streaming
                    else self.train_dataset.select(range(self.max_train_samples))
                )
        if self.do_eval:
            if "validation" not in dataset:
                raise ValueError("Evaluating requires a validation dataset")
            self.eval_dataset = dataset["validation"]
            if self.max_eval_samples is not None:
                self.eval_dataset = (
                    self.eval_dataset.take(self.max_eval_samples)
                    if self.streaming
                    else self.eval_dataset.select(range(self.max_eval_samples))
                )

    def preprocess(self, tokenizer, decoder_start_token_id, normalize_text, max_length):
        if self.streaming:
            # we need to shuffle early in streaming mode
            if hasattr(self, "train_dataset"):
                self.train_dataset = self.train_dataset.shuffle(1000, self.seed_dataset)
        else:
            # prepare rng for later shuffling
            if self.seed_dataset is None:
                self.seed_dataset = np.random.get_state()[1][0]
            self.rng_dataset = jax.random.PRNGKey(self.seed_dataset)

        # normalize text
        if normalize_text:
            text_normalizer = TextNormalizer()
            partial_normalize_function = partial(
                normalize_function,
                text_column=self.text_column,
                text_normalizer=text_normalizer,
            )
            for ds in ["train_dataset", "eval_dataset"]:
                if hasattr(self, ds):
                    setattr(
                        self,
                        ds,
                        (
                            getattr(self, ds).map(partial_normalize_function)
                            if self.streaming
                            else getattr(self, ds).map(
                                partial_normalize_function,
                                num_proc=self.preprocessing_num_workers,
                                load_from_cache_file=not self.overwrite_cache,
                                desc="Normalizing datasets",
                            )
                        ),
                    )

        # preprocess
        partial_preprocess_function = partial(
            preprocess_function,
            tokenizer=tokenizer,
            text_column=self.text_column,
            encoding_column=self.encoding_column,
            max_length=max_length,
            decoder_start_token_id=decoder_start_token_id,
        )
        for ds in ["train_dataset", "eval_dataset"]:
            if hasattr(self, ds):
                setattr(
                    self,
                    ds,
                    (
                        getattr(self, ds).map(
                            partial_preprocess_function,
                            batched=True,
                        )
                        if self.streaming
                        else getattr(self, ds).map(
                            partial_preprocess_function,
                            batched=True,
                            remove_columns=getattr(ds, "column_names"),
                            num_proc=self.preprocessing_num_workers,
                            load_from_cache_file=not self.overwrite_cache,
                            desc="Preprocessing datasets",
                        )
                    ),
                )

    def dataloader(self, split, batch_size, epoch=None):
        def _dataloader_datasets_non_streaming(
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

            batch_idx = batch_idx[
                : steps_per_epoch * batch_size
            ]  # Skip incomplete batch.
            batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

            for idx in batch_idx:
                batch = dataset[idx]
                batch = {k: jnp.array(v) for k, v in batch.items()}
                batch = shard(batch)
                yield batch

        def _dataloader_datasets_streaming(
            dataset: Dataset, batch_size: int, epoch: int
        ):
            # epoch is only use for multi-host
            keys = ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
            batch = {k: [] for k in keys}
            first_loop = True
            while self.multi_hosts or first_loop:
                # in multi-host, we run forever (no epoch) as hosts need to stop
                # at the same time and we don't know how much data is on each host
                if not first_loop:
                    # multi-host setting, we reshuffle shards
                    epoch += 1
                    dataset.set_epoch(epoch)
                for item in dataset:
                    for k, v in item.items():
                        batch[k].append(v)
                    if len(batch[keys[0]]) == batch_size:
                        batch = {k: jnp.array(v) for k, v in batch.items()}
                        batch = shard(batch)
                        yield batch
                        batch = {k: [] for k in keys}
                first_loop = False

        if split == "train":
            ds = self.train_dataset
        elif split == "eval":
            ds = self.eval_dataset
        else:
            raise ValueError(f'split must be "train" or "eval", got {split}')

        if self.streaming:
            if split == "train":
                ds.set_epoch(epoch)
            return _dataloader_datasets_streaming(ds, batch_size, epoch)
        else:
            if split == "train":
                self.rng_dataset, input_rng = jax.random.split(self.rng_dataset)
            return _dataloader_datasets_non_streaming(ds, batch_size, input_rng)

    @property
    def length(self):
        len_train_dataset, len_eval_dataset = None, None
        if self.streaming:
            # we don't know the length, let's just assume max_samples if defined
            if self.max_train_samples is not None:
                len_train_dataset = self.max_train_samples
            if self.max_eval_samples is not None:
                len_eval_dataset = self.max_eval_samples
        else:
            len_train_dataset = (
                len(self.train_dataset) if hasattr(self, "train_dataset") else None
            )
            len_eval_dataset = (
                len(self.eval_dataset) if hasattr(self, "eval_dataset") else None
            )
        return len_train_dataset, len_eval_dataset


def shift_tokens_right(input_ids: np.array, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


def normalize_function(example, text_column, text_normalizer):
    example[text_column] = text_normalizer(example[text_column])
    return example


def preprocess_function(
    examples,
    tokenizer,
    text_column,
    encoding_column,
    max_length,
    decoder_start_token_id,
):
    inputs = examples[text_column]
    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
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
    decoder_input_ids = shift_tokens_right(labels, decoder_start_token_id)
    model_inputs["decoder_input_ids"] = decoder_input_ids

    return model_inputs
