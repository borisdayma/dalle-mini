---
title: DALL·E mini
emoji: 🥑
colorFrom: yellow
colorTo: green
sdk: streamlit
app_file: app/app.py
pinned: false
---

# DALL·E Mini

_Generate images from a text prompt_

<img src="img/logo.png" width="200">

Our logo was generated with DALL·E mini using the prompt "logo of an armchair in the shape of an avocado".

You can create your own pictures with [the demo](https://huggingface.co/spaces/flax-community/dalle-mini).

## How does it work?

Refer to [our report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA).

## Development

### Dependencies Installation

The root folder and associated [`requirements.txt`](./requirements.txt) is only for the app.

For development, use [`dev/requirements.txt`](dev/requirements.txt) or [`dev/environment.yaml`](dev/environment.yaml).

### Training of VQGAN

The VQGAN was trained using [taming-transformers](https://github.com/CompVis/taming-transformers).

We recommend using the latest version available.

### Conversion of VQGAN to JAX

Use [patil-suraj/vqgan-jax](https://github.com/patil-suraj/vqgan-jax).

### Training of Seq2Seq

Refer to [`dev/seq2seq`](dev/seq2seq) folder.

You can also adjust the [sweep configuration file](https://docs.wandb.ai/guides/sweeps) if you need to perform a hyperparameter search.

### Inference Pipeline

To generate sample predictions and understand the inference pipeline step by step, refer to [`dev/inference/inference_pipeline.ipynb`](dev/inference/inference_pipeline.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/dev/inference/inference_pipeline.ipynb)

## FAQ

### Where to find the latest models?

Trained models are on 🤗 Model Hub:

- [VQGAN-f16-16384](https://huggingface.co/flax-community/vqgan_f16_16384) for encoding/decoding images
- [DALL·E mini](https://huggingface.co/flax-community/dalle-mini) for generating images from a text prompt

### Where does the logo come from?

The "armchair in the shape of an avocado" was used by OpenAI when releasing DALL·E to illustrate the model's capabilities. Having successful predictions on this prompt represents a big milestone to us.

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

## Citing DALL·E mini

If you find DALL·E mini useful in your research or wish to refer, please use the following BibTeX entry.

```
@misc{Dayma_DALL·E_Mini_2021,
author = {Dayma, Boris and Patil, Suraj and Cuenca, Pedro and Saifullah, Khalid and Abraham, Tanishq and Lê Khắc, Phúc and Melas, Luke and Ghosh, Ritobrata},
doi = {10.5281/zenodo.5146400},
month = {7},
title = {DALL·E Mini},
url = {https://github.com/borisdayma/dalle-mini},
year = {2021}
}
```

## References

```
@misc{ramesh2021zeroshot,
      title={Zero-Shot Text-to-Image Generation}, 
      author={Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
      year={2021},
      eprint={2102.12092},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{esser2021taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2021},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{lewis2019bart,
      title={BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension}, 
      author={Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and Abdelrahman Mohamed and Omer Levy and Ves Stoyanov and Luke Zettlemoyer},
      year={2019},
      eprint={1910.13461},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```