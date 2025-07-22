# Axolotl CUDA128 Custom Docker Image

```
docker build -t axolotl-cuda128-base ./axolotl-cuda128-docker/

docker run --privileged --gpus '"all"' --shm-size 10g --rm -it \
  --name axolotl --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="${PWD}",target=/workspace \
  -v ${HF_HOME}:/root/.cache/huggingface \
  -e "CUDA_VISIBLE_DEVICES=5" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "WANDB_API_KEY=$WANDB_API_KEY" \
  axolotl-cuda128-base
```