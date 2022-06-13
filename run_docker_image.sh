#!/bin/bash

# This script is used to run the docker image. Change or remove GPU flag if you dont have nvidia-docker or the needed GPUs
docker run --rm --name dallemini -it -p 8888:8888  --gpus all  -v "${PWD}":/workspace dalle-mini:latest
