# Running Dalle-mini With Docker

This folder contains the Dockerfile needed to build a Docker image that can easily run Dalle-mini.

## Inference

Steps to run inference with Dalle-mini are as follows:

1. Build the docker image with ```dalle-mini/Docker/build_image.sh```
2. Run the container with ```dalle-mini/run_docker_image```
3. Navigate to ```/workspace/tools/inference/``` and run ```run_infer_notebook.sh```
4. Click the Jupyter Notebook link and run through the notebook.
