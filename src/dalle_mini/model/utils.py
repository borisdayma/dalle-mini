import os
import tempfile
from pathlib import Path
from typing import Optional
import wandb


class PretrainedFromWandbMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        api_key: Optional[str] = None,
        *model_args,
        **kwargs
    ):
        """
        Initializes from a wandb artifact or delegates loading to the superclass.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if ":" in pretrained_model_name_or_path and not os.path.isdir(
                pretrained_model_name_or_path
            ):
                # wandb artifact
                if wandb.run is not None:
                    artifact = wandb.run.use_artifact(pretrained_model_name_or_path)
                else:
                    # pass api_key if set, otherwise a prompt will display to enter the key
                    artifact = wandb.Api(api_key=api_key).artifact(
                        pretrained_model_name_or_path
                    )
                pretrained_model_name_or_path = artifact.download(tmp_dir)

            return super(PretrainedFromWandbMixin, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
