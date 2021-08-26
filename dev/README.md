# Development Instructions for TPU

## Setup

- Apply to the [TRC program](https://sites.research.google/trc/) for free TPU credits if you're elligible.
- Follow the [Cloud TPU VM User's Guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm) to set up gcloud.
- Verify `gcloud config list`, in particular account, project & zone.
- Create a TPU VM per the guide and connect to it.

When needing a larger disk:

- Create a balanced persistent disk (SSD, so pricier than default HDD but much faster): `gcloud compute disks create DISK_NAME --size SIZE_IN_GB --type pd-balanced`
- Attach the disk to your instance by adding `--data-disk source=REF` per ["Adding a persistent disk to a TPU VM" guide](https://cloud.google.com/tpu/docs/setup-persistent-disk), eg `gcloud alpha compute tpus tpu-vm create INSTANCE_NAME --accelerator-type=v3-8 --version=v2-alpha --data-disk source=projects/tpu-toys/zones/europe-west4-a/disks/DISK_NAME`
- Format the partition as described in the guide.
- Make sure to set up automatic remount of disk at restart.

## Connect VS Code

- Find external IP in the UI or with `gcloud alpha compute tpus tpu-vm describe INSTANCE_NAME`
- Verify you can connect in terminal with `ssh EXTERNAL_IP -i ~/.ssh/google_compute_engine`
- Add the same command as ssh host in VS Code.
- Check config file

  ```
  Host INSTANCE_NAME
    HostName EXTERNAL_IP
    IdentityFile ~/.ssh/google_compute_engine
  ```

## Environment configuration

### Use virtual environments (optional)

We recommend using virtual environments (such as conda, venv or pyenv-virtualenv).

If you want to use `pyenv` and `pyenv-virtualenv`:

- Installation

  - [Set up build environment](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)
  - Use [pyenv-installer](https://github.com/pyenv/pyenv-installer): `curl https://pyenv.run | bash`
  - bash set-up:

    ```bash
    echo '\n'\
        '# pyenv setup \n'\
        'export PYENV_ROOT="$HOME/.pyenv" \n'\
        'export PATH="$PYENV_ROOT/bin:$PATH" \n'\
        'eval "$(pyenv init --path)" \n'\
        'eval "$(pyenv init -)" \n'\
        'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    ```

- Usage

  - Install a python version: `pyenv install X.X.X`
  - Create a virtual environment: `pyenv virtualenv 3.9.6 dalle_env`
  - Activate: `pyenv activate dalle_env`

    Note: you can auto-activate your environment at a location with `echo dalle_env >> .python-version`

### Tools

- Git

  - `git config --global user.email "name@domain.com"
  - `git config --global user.name "First Last"

- Github CLI

  - See [installation instructions](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)
  - `gh auth login`

- Direnv

  - Install direnv: `sudo apt-get update && sudo apt-get install direnv`
  - bash set-up:

    ```bash
    echo -e '\n'\
        '# direnv setup \n'\
        'eval "$(direnv hook bash)" \n' >> ~/.bashrc
    ```

### Set up repo

- Clone repo: `gh repo clone borisdayma/dalle-mini`
- If using `pyenv-virtualenv`, auto-activate env: `echo dalle_env >> .python-version`

## Environment

- Install the following (use it later to update our dev requirements.txt)

```
requests
pillow
jupyterlab
ipywidgets

-e ../datasets[streaming]
-e ../transformers
-e ../webdataset

# JAX
--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]>=0.2.16
flax
```

- `transformers-cli login`

---

- set `HF_HOME="/mnt/disks/persist/cache/huggingface"` in `/etc/environment` and ensure you have required permissions, then restart.

## Working with datasets or models

- Install [Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation)
- Clone a dataset without large files: `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/.../...`
- Use a local [credential store](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage) for caching credentials
- Track specific extentions: `git lfs track "*.ext"`
- See files tracked with LFS with `git lfs ls-files`
