---
title: DALL·E mini
emoji: 🥑
colorFrom: red
colorTo: purple
sdk: streamlit
app_file: app/app.py
pinned: false
---

# DALL·E Mini

_Generate images from a text prompt_

<img src="img/logo.png" width="200">

Our logo was generated with DALL·E mini using the prompt "logo of an armchair in the shape of an avocado".

You can create your own pictures with [the demo](https://huggingface.co/spaces/flax-community/dalle-mini) (temporarily in beta on Huging Face Spaces but soon to be open to all).

## How does it work?

Refer to [our report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA).

## Development

This section is for the adventurous people wanting to look into the code.

### Dependencies Installation

The root folder and associated `requirements.txt` is only for the app.

You will find necessary requirements in each sub-section.

You should create a new python virtual environment and install the project dependencies inside the virtual env. You need to use the `-f` (`--find-links`) option for `pip` to be able to find the appropriate `libtpu` required for the TPU hardware.

Adapt the installation to your own hardware and follow library installation instructions.

```
$ pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

If you use `conda`, you can create the virtual env and install everything using: `conda env update -f environments.yaml`

### Training of VQGAN

The VQGAN was trained using [taming-transformers](https://github.com/CompVis/taming-transformers).

We recommend using the latest version available.

### Conversion of VQGAN to JAX

Use [patil-suraj/vqgan-jax](https://github.com/patil-suraj/vqgan-jax).

### Training of Seq2Seq

Refer to `dev/seq2seq` folder.

You can also adjust the [sweep configuration file](https://docs.wandb.ai/guides/sweeps) if you need to perform a hyperparameter search.

### Inference

Refer to `dev/notebooks/demo`.

## Authors

- [Boris Dayma](https://github.com/borisdayma)
- [Suraj Patil](https://github.com/patil-suraj)
- [Pedro Cuenca](https://github.com/pcuenca)
- [Khalid Saifullah](https://github.com/khalidsaifullaah)
- [Tanishq Abraham](https://github.com/tmabraham)
- [Phúc Lê Khắc](https://github.com/lkhphuc)
- [Luke Melas](https://github.com/lukemelas)
- [Ritobrata Ghosh](https://github.com/ghosh-r)

## Acknowledgements

- 🤗 Hugging Face for organizing [the FLAX/JAX community week](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects)
- Google Cloud team for providing access to TPU's
