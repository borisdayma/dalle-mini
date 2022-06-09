#!/bin/bash

# This script is used to run the docker image. Change or remove GPU flag if you dont have nvidia-docker or the needed GPUs
current_dir=$(pwd)
docker run -it -p 8888:8888  --gpus all  -v $current_dir:/workspace dalle-mini:latest
