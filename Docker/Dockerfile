FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG device_type=auto

RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN if [ "$device_type" = "auto" ]; then \
  pip install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_releases.html; fi

RUN if [ "$device_type" = "nocuda" ]; then \
  pip install --upgrade "jax[nocuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html; fi

RUN if [ "$device_type" = "cuda" ]; then \
  pip install --upgrade "jaxlib[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; fi

RUN pip install jupyter

WORKDIR /workspace