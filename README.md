## DALL-E Mini - Generate image from text

## Tentative Strategy of training (proposed by Luke and Suraj)

### Data: 
* [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) Dataset (already loaded and preprocessed in TPU VM by Luke).
* [YFCC100M Subset](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md)
* [Coneptual Captions 3M](https://github.com/google-research-datasets/conceptual-captions)

### Architecture: 
  * Use the Taming Transformers VQ-GAN (with 16384 tokens)
  * Use a seq2seq (language encoder --> image decoder) model with a pretrained non-autoregressive encoder (e.g. BERT) and an autoregressive decoder (like GPT). 

### Remaining Architecture Questions: 
  * Whether to freeze the text encoder?
  * Whether to finetune the VQ-GAN?
  * Which text encoder to use (e.g. BERT, RoBERTa, etc.)?
  * Hyperparameter choices for the decoder (e.g. positional embedding, initialization, etc.)

## TODO

* experiment with flax/jax and setup of the TPU instance that we should get shortly
* work on dataset loading - [see suggested datasets](https://discuss.huggingface.co/t/dall-e-mini-version/7324/4)
* Optionally create the OpenAI YFCC100M subset (see [this post](https://discuss.huggingface.co/t/dall-e-mini-version/7324/30?u=boris))
* work on text/image encoding
* concatenate inputs (not sure if we need fixed length for text or use a special token separating text & image)
* adapt training script
* create inference function
* integrate CLIP for better results (only if we have the time)
* work on a demo (streamlit or colab or maybe just HF widget)
* document (set up repo on model hub per instructions, start on README writeup…)
* help with coordinating activities & progress


## Dependencies Installation
You should create a new python virtual environment and install the project dependencies inside the virtual env. You need to use the `-f` (`--find-links`) option for `pip` to be able to find the appropriate `libtpu` required for the TPU hardware:

```
$ pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

If you use `conda`, you can create the virtual env and install everything using: `conda env update -f environments.yaml`
