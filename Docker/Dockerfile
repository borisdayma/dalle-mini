FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG device_type=auto

RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN if [ "$device_type" = "auto" ]; then \
  pip install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \
  pip install -q \
  git+https://github.com/borisdayma/dalle-mini.git \
  git+https://github.com/patil-suraj/vqgan-jax.git; fi

RUN if [ "$device_type" = "cpu" ]; then \
  pip install --upgrade https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.3.10-cp38-none-manylinux2014_x86_64.whl \
  pip install -q \
  git+https://github.com/borisdayma/dalle-mini.git \
  git+https://github.com/patil-suraj/vqgan-jax.git; fi

RUN if [ "$device_type" = "gpu" ]; then \
  pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.10+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl \
  pip install -q \
  git+https://github.com/borisdayma/dalle-mini.git \
  git+https://github.com/patil-suraj/vqgan-jax.git; fi

RUN pip install jupyter

WORKDIR /workspace