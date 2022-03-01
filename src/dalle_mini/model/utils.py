import os
import tempfile
from pathlib import Path

import wandb


class PretrainedFromWandbMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Initializes from a wandb artifact, google bucket path or delegates loading to the superclass.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if (
                ":" in pretrained_model_name_or_path
                and not os.path.isdir(pretrained_model_name_or_path)
                and not pretrained_model_name_or_path.startswith("gs")
            ):
                # wandb artifact
                if wandb.run is not None:
                    artifact = wandb.run.use_artifact(pretrained_model_name_or_path)
                else:
                    artifact = wandb.Api().artifact(pretrained_model_name_or_path)
                pretrained_model_name_or_path = artifact.download(tmp_dir)

            return super(PretrainedFromWandbMixin, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )


def copy_blobs(source_path, dest_path):
    assert source_path.startswith("gs://")
    from google.cloud import storage

    bucket_path = Path(source_path[5:])
    bucket, dir_path = str(bucket_path).split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket)
    blobs = client.list_blobs(bucket, prefix=f"{dir_path}/")
    for blob in blobs:
        dest_name = str(Path(dest_path) / Path(blob.name).name)
        blob.download_to_filename(dest_name)
