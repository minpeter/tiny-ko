# tinyko - A Tiny LLM for korean Language

## installation

```shell
# https://github.com/astral-sh/uv/issues/6437#issuecomment-2535324784
uv sync --no-install-package flash-attn
uv sync --no-build-isolation
```

## Usage

```shell
uv run 00-tknz.py
uv run 01-preprocess.py
uv run accelerate launch 02-train.py --hf_model_id your-hf/model-id
```


## ckpt

https://huggingface.co/minpeter/tiny-ko-187m-base-250718  
https://huggingface.co/minpeter/tiny-ko-124m-base-muon  
https://huggingface.co/minpeter/tiny-ko-20m-base-en  
.... 


## Model Evaluation (lm-eval)
https://wandb.ai/kasfiekfs-e/lm-eval-harness-integration/workspace  


## Other similar projects
https://github.com/huggingface/smollm/tree/main/text/pretraining  
https://github.com/jzhang38/TinyLlama  
https://github.com/SmallDoges/small-doge  
https://github.com/keeeeenw/MicroLlama  
https://github.com/karpathy/nanoGPT  
[[Model] Very, very small things Collections](https://huggingface.co/collections/minpeter/model-very-very-small-things-68660f5eaa427e3d37b8ca8a)
 
