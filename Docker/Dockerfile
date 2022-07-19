# Use the tensorflow image as base for CUDA compatibility
FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
		python3 \
		python3-pip \
		&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel jupyter
RUN pip install --use-deprecated=html5lib jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install -q git+https://github.com/borisdayma/dalle-mini.git
RUN pip install -q git+https://github.com/patil-suraj/vqgan-jax.git

WORKDIR /workspace
