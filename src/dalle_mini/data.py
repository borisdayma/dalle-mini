import random
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from braceexpand import braceexpand
from datasets import Dataset, load_dataset

from .model.text import TextNormalizer


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
    blank_caption_prob: float = 0.0
    clip_score_column: str = "clip_score"
    min_clip_score: float = None
    max_clip_score: float = None
    filter_column: str = None
    filter_value: str = None
    train_dataset: Dataset = field(init=False)
    eval_dataset: Dataset = field(init=False)
    rng_dataset: jnp.ndarray = field(init=False)
    multi_hosts: bool = field(init=False)

    def __post_init__(self):
        if self.seed_dataset is None:
            # create a random seed
            self.seed_dataset = random.randint(0, 2**32 - 1)
        # set numpy rng
        self.np_rng = np.random.default_rng(self.seed_dataset)
        self.multi_hosts = jax.process_count() > 1
        # feed blank captions only in streaming mode for now
        # otherwise dataset could be cached with same blanked captions
        if self.blank_caption_prob:
            assert (
                self.streaming is True
            ), "blank_caption_prob can only be used in streaming mode"
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

    def preprocess(self, tokenizer, config):
        # get required config variables
        decoder_start_token_id = config.decoder_start_token_id
        normalize_text = config.normalize_text
        max_length = config.max_text_length

        if self.streaming:
            # we need to shuffle early in streaming mode
            if hasattr(self, "train_dataset"):
                self.train_dataset = self.train_dataset.shuffle(
                    buffer_size=5000, seed=self.seed_dataset
                )
        else:
            self.rng_dataset = jax.random.PRNGKey(self.seed_dataset)

        # filter data
        partial_filter_function = partial(
            filter_function,
            filter_column=self.filter_column,
            filter_value=self.filter_value,
            clip_score_column=self.clip_score_column,
            min_clip_score=self.min_clip_score,
            max_clip_score=self.max_clip_score,
        )
        for ds in ["train_dataset", "eval_dataset"]:
            if hasattr(self, ds):
                setattr(
                    self,
                    ds,
                    (
                        getattr(self, ds).filter(partial_filter_function)
                        if self.streaming
                        else getattr(self, ds).filter(
                            partial_filter_function,
                            num_proc=self.preprocessing_num_workers,
                            load_from_cache_file=not self.overwrite_cache,
                            desc="Filtering datasets",
                        )
                    ),
                )

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

        # blank captions
        if self.blank_caption_prob:
            partial_blank_caption_function = partial(
                blank_caption_function,
                text_column=self.text_column,
                blank_caption_prob=self.blank_caption_prob,
                rng=self.np_rng,
            )
            if hasattr(self, "train_dataset"):
                self.train_dataset = (
                    self.train_dataset.map(partial_blank_caption_function)
                    if self.streaming
                    else self.train_dataset.map(
                        partial_blank_caption_function,
                        num_proc=None
                        if self.seed_dataset
                        else self.preprocessing_num_workers,
                        load_from_cache_file=False,
                        desc="Blanking some captions",
                    )
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
                            remove_columns=[
                                self.text_column,
                                self.encoding_column,
                            ],
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
            rng: jax.random.PRNGKey = None,
        ):
            """
            Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
            Shuffle batches if rng is set.
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
                yield batch

        def _dataloader_datasets_streaming(
            dataset: Dataset,
            epoch: int,
        ):
            keys = ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
            batch = {k: [] for k in keys}
            first_loop = True  # stop after one loop in some cases
            while (self.multi_hosts and split == "train") or first_loop:
                # in multi-host, we run forever (no epoch) as hosts need to stop
                # at the same time and training data may not be split equally
                # For validation data we put the entire batch on each host and then
                # keep only the one specific to each host (could be improved but not necessary)
                if epoch is not None:
                    assert split == "train"
                    # reshuffle training data at each epoch
                    dataset.set_epoch(epoch)
                    epoch += 1
                for item in dataset:
                    for k in keys:
                        batch[k].append(item[k])
                    if len(batch[keys[0]]) == batch_size:
                        batch = {k: jnp.array(v) for k, v in batch.items()}
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
            return _dataloader_datasets_streaming(ds, epoch)
        else:
            if split == "train":
                self.rng_dataset, input_rng = jax.random.split(self.rng_dataset)
            return _dataloader_datasets_non_streaming(ds, input_rng)

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


def blank_caption_function(example, text_column, blank_caption_prob, rng=None):
    if (
        blank_caption_prob
        and (rng.random() if rng is not None else np.random.random())
        < blank_caption_prob
    ):
        example[text_column] = ""
    return example


def normalize_function(example, text_column, text_normalizer):
    example[text_column] = text_normalizer(example[text_column])
    return example


def filter_function(
    example,
    min_clip_score,
    max_clip_score,
    clip_score_column,
    filter_column,
    filter_value,
):
    if min_clip_score is not None and example[clip_score_column] < min_clip_score:
        return False
    if max_clip_score is not None and example[clip_score_column] > max_clip_score:
        return False
    if filter_column is not None and example[filter_column] != filter_value:
        return False
    return True


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
