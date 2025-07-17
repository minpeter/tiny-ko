# tinyko - A Tiny LLM for korean Language

## installation

```shell
uv sync --no-build-isolation-package flash-attn
```

## Usage

```shell
uv run 00-tknz.py
uv run 01-preprocess.py
uv run accelerate launch 02-train.py --hf_model_id your-hf/model-id
```
