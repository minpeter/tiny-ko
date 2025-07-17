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
