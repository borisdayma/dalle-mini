## DALL-E Mini - Generate image from text

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
