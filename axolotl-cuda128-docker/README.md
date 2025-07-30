# Axolotl CUDA128 Custom Docker Image

You can create and run an Ubuntu Docker image for CUDA 12.8.1 using the following commands.
This resolves an unknown bug (axolotl.cli not found) that exists in the official release.

```
docker build -t axolotl-cuda128-base ./axolotl-cuda128-docker/

docker run --gpus '"all"' --rm -it axolotl-cuda128-base
```

Or you can use the pre-built `minpeter/axolotl-cuda128-base` image.

```
docker pull minpeter/axolotl-cuda128-base:latest

docker run --privileged --gpus '"all"' --shm-size 10g --rm -it \
  --name axolotl --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="${PWD}",target=/workspace \
  -v ${HF_HOME}:/root/.cache/huggingface \
  -e "CUDA_VISIBLE_DEVICES=5" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "WANDB_API_KEY=$WANDB_API_KEY" \
  minpeter/axolotl-cuda128-base:latest
```

## CUDA 12.9.1

```
docker build \
  --build-arg CUDA_VERSION=12.9.1 \
  --build-arg CUDA=129 \
  --build-arg TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0 10.0+PTX" \
  -t axolotl-cuda129-base \
  ./axolotl-cuda128-docker/
```