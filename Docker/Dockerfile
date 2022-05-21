FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install git -y
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install -q git+https://github.com/borisdayma/dalle-mini.git
RUN pip install -q git+https://github.com/patil-suraj/vqgan-jax.git
WORKDIR /workspace
