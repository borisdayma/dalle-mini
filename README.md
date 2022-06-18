# DALL·E Mini

[![Join us on Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/xBPBXfcFHd)

_Generate images from a text prompt_

<img src="https://github.com/borisdayma/dalle-mini/raw/main/img/logo.png" width="200">

Our logo was generated with DALL·E mini using the prompt "logo of an armchair in the shape of an avocado".

## How to use it?

There are several ways to use DALL·E mini to create your own images:

* use [the official DALL·E Mini demo](https://huggingface.co/spaces/dalle-mini/dalle-mini)

* experiment with the pipeline step by step through our [`inference pipeline notebook`](tools/inference/inference_pipeline.ipynb)

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb)

You can also use these great projects from the community:

* spin off your own app with [DALL-E Playground repository](https://github.com/saharmor/dalle-playground) (thanks [Sahar](https://twitter.com/theaievangelist))

* try [DALL·E Flow](https://github.com/jina-ai/dalle-flow) project for generating, diffusion, and upscaling in a Human-in-the-Loop workflow (thanks [Han Xiao](https://github.com/hanxiao))

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb)

## How does it work?

Refer to [our report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini-Generate-images-from-any-text-prompt--VmlldzoyMDE4NDAy).

## Contributing

Join the community on the [LAION Discord](https://discord.gg/xBPBXfcFHd).
Any contribution is welcome, from reporting issues to proposing fixes/improvements or testing the model with cool prompts!

## Development

### Dependencies Installation

For inference only, use `pip install git+https://github.com/borisdayma/dalle-mini.git`.

For development, clone the repo and use `pip install -e ".[dev]"`.
Before making a PR, check style with `make style`.

### Training of DALL·E mini

Use [`tools/train/train.py`](tools/train/train.py).

You can also adjust the [sweep configuration file](https://docs.wandb.ai/guides/sweeps) if you need to perform a hyperparameter search.

## FAQ

### Where to find the latest models?

Trained models are on 🤗 Model Hub:

* [VQGAN-f16-16384](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384) for encoding/decoding images
* [DALL·E mini](https://huggingface.co/flax-community/dalle-mini) for generating images from a text prompt

### Where does the logo come from?

The "armchair in the shape of an avocado" was used by OpenAI when releasing DALL·E to illustrate the model's capabilities. Having successful predictions on this prompt represents a big milestone for us.

## Acknowledgements

* 🤗 Hugging Face for organizing [the FLAX/JAX community week](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects)
* Google [TPU Research Cloud (TRC) program](https://sites.research.google/trc/) for providing computing resources
* [Weights & Biases](https://wandb.com/) for providing the infrastructure for experiment tracking and model management

## Authors & Contributors

DALL·E mini was initially developed by:

* [Boris Dayma](https://github.com/borisdayma)
* [Suraj Patil](https://github.com/patil-suraj)
* [Pedro Cuenca](https://github.com/pcuenca)
* [Khalid Saifullah](https://github.com/khalidsaifullaah)
* [Tanishq Abraham](https://github.com/tmabraham)
* [Phúc Lê Khắc](https://github.com/lkhphuc)
* [Luke Melas](https://github.com/lukemelas)
* [Ritobrata Ghosh](https://github.com/ghosh-r)

Many thanks to the people who helped make it better:

* the [DALLE-Pytorch](https://discord.gg/xBPBXfcFHd) and [EleutherAI](https://www.eleuther.ai/) communities for testing and exchanging cool ideas
* [Rohan Anil](https://github.com/rohan-anil) for adding Distributed Shampoo optimizer and always giving great suggestions
* [Phil Wang](https://github.com/lucidrains) has provided a lot of cool implementations of transformer variants and gives interesting insights with [x-transformers](https://github.com/lucidrains/x-transformers)
* [Katherine Crowson](https://github.com/crowsonkb) for [super conditioning](https://twitter.com/RiversHaveWings/status/1478093658716966912)
* the [Gradio team](https://gradio.app/) made an amazing UI for our app

## Citing DALL·E mini

If you find DALL·E mini useful in your research or wish to refer, please use the following BibTeX entry.

```text
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

Original DALL·E from "[Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)" with image quantization from "[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)".

Image encoder from "[Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841v2)".

Sequence to sequence model based on "[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461v1)" with implementation of a few variants:

* "[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)"
* "[Deepnet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)"
* "[NormFormer: Improved Transformer Pretraining with Extra Normalization](https://arxiv.org/abs/2110.09456)"
* "[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)"
* "[CogView: Mastering Text-to-Image Generation via Transformers](https://arxiv.org/abs/2105.13290v2)"
* "[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)"
* "[Sinkformers: Transformers with Doubly Stochastic Attention](https://arxiv.org/abs/2110.11773)"

Main optimizer (Distributed Shampoo) from "[Scalable Second Order Optimization for Deep Learning](https://arxiv.org/abs/2002.09018)".

### Citations

```text
@misc{
  title={Zero-Shot Text-to-Image Generation}, 
  author={Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
  year={2021},
  eprint={2102.12092},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

```text
@misc{
  title={Learning Transferable Visual Models From Natural Language Supervision}, 
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  year={2021},
  eprint={2103.00020},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

```text
@misc{
  title={Taming Transformers for High-Resolution Image Synthesis}, 
  author={Patrick Esser and Robin Rombach and Björn Ommer},
  year={2021},
  eprint={2012.09841},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

```text
@misc{
  title={BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension}, 
  author={Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and Abdelrahman Mohamed and Omer Levy and Ves Stoyanov and Luke Zettlemoyer},
  year={2019},
  eprint={1910.13461},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

```text
@misc{
  title={Scalable Second Order Optimization for Deep Learning},
  author={Rohan Anil and Vineet Gupta and Tomer Koren and Kevin Regan and Yoram Singer},
  year={2021},
  eprint={2002.09018},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

```text
@misc{
  title={GLU Variants Improve Transformer},
  author={Noam Shazeer},
  year={2020},
  url={https://arxiv.org/abs/2002.05202}    
}
```

```text
 @misc{
  title={DeepNet: Scaling transformers to 1,000 layers},
  author={Wang, Hongyu and Ma, Shuming and Dong, Li and Huang, Shaohan and Zhang, Dongdong and Wei, Furu},
  year={2022},
  eprint={2203.00555}
  archivePrefix={arXiv},
  primaryClass={cs.LG}
} 
```

```text
@misc{
  title={NormFormer: Improved Transformer Pretraining with Extra Normalization},
  author={Sam Shleifer and Jason Weston and Myle Ott},
  year={2021},
  eprint={2110.09456},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

```text
@inproceedings{
  title={Swin Transformer V2: Scaling Up Capacity and Resolution}, 
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

```text
@misc{
  title = {CogView: Mastering Text-to-Image Generation via Transformers},
  author = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
  year = {2021},
  eprint = {2105.13290},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

```text
@misc{
  title = {Root Mean Square Layer Normalization},
  author = {Biao Zhang and Rico Sennrich},
  year = {2019},
  eprint = {1910.07467},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

```text
@misc{
  title = {Sinkformers: Transformers with Doubly Stochastic Attention},
  url = {https://arxiv.org/abs/2110.11773},
  author = {Sander, Michael E. and Ablin, Pierre and Blondel, Mathieu and Peyré, Gabriel},
  publisher = {arXiv},
  year = {2021},
}
```

```text
@misc{
  title = {Smooth activations and reproducibility in deep networks},
  url = {https://arxiv.org/abs/2010.09931},
  author = {Shamir, Gil I. and Lin, Dong and Coviello, Lorenzo},
  publisher = {arXiv},
  year = {2020},
}
```
