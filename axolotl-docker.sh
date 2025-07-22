docker run --privileged --gpus '"all"' --shm-size 10g --rm -it \
  --name axolotl --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="${PWD}",target=/workspace/axolotl \
  -v ${HF_HOME}:/root/.cache/huggingface \
  -e "CUDA_VISIBLE_DEVICES=5" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "WANDB_API_KEY=$WANDB_API_KEY" \
  axolotlai/axolotl:main-py3.11-cu128-2.7.1


ARG CUDA_VERSION="12.8.1"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG MAX_JOBS=4

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION AS base-builder


docker pull nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04


docker run --privileged --gpus '"all"' --shm-size 10g --rm -it \
  --name axolotl --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="${PWD}",target=/workspace/axolotl \
  -v ${HF_HOME}:/root/.cache/huggingface \
  -e "CUDA_VISIBLE_DEVICES=5" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "WANDB_API_KEY=$WANDB_API_KEY" \
  nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04


uv pip install packaging setuptools wheel
uv pip install torch
uv pip install awscli pydantic

pip3 install --no-build-isolation 'axolotl[flash-attn,deepspeed]'



## CUDA 12.8.1 이미지로, miniconda 설치해서, 그 위에서 python 3.11 설치하고, 그 위에서 axolotl 올리면!
# 동작함.