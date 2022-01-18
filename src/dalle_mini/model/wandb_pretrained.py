import os
import wandb


class PretrainedFromWandbMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Initializes from a wandb artifact, or delegates loading to the superclass.
        """
        if ":" in pretrained_model_name_or_path and not os.path.isdir(
            pretrained_model_name_or_path
        ):
            # wandb artifact
            artifact = wandb.Api().artifact(pretrained_model_name_or_path)
            pretrained_model_name_or_path = artifact.download()

        return super(PretrainedFromWandbMixin, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
