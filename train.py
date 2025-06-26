# RUN COMMAND: time uv run accelerate launch train.py

import os
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

ds_kr = load_dataset("minpeter/tiny-ko-corpus", split="train[:50_000]")

# >>> en dataset >>>
cosmopedia = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    data_files=[f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(21)],
    split="train[:50_000]",
)
fineweb = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    data_files=[f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet" for i in range(21)],
    split="train[:50_000]",
)
cosmopedia_text = cosmopedia.remove_columns(
    [col for col in cosmopedia.column_names if col != "text"]
)
fineweb_text = fineweb.remove_columns(
    [col for col in fineweb.column_names if col != "text"]
)
ds_en = concatenate_datasets([cosmopedia_text, fineweb_text])
# <<< en dataset <<<

ds = concatenate_datasets([ds_kr, ds_en])
ds = ds.train_test_split(test_size=0.001, shuffle=True, seed=5768112)
print(ds)

context_length = 2048
max_cpu_count = int(os.cpu_count() / 3) or 1

tokenizer = AutoTokenizer.from_pretrained("./tknz/tiny-ko-tokenizer")
tokenizer.model_max_length = context_length

try:
    print(f"사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
except AttributeError as e:
    print(e)


def tokenize_with_eos(examples):
    # 각 텍스트의 끝에 EOS 토큰 추가
    texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
    # truncation=False, padding=False로 설정하여 원본 길이 그대로 토큰화
    return tokenizer(texts_with_eos, truncation=False, padding=False)


print("\nEOS 토큰 추가와 토큰화를 동시에 진행 중...")
tokenized_ds = ds.map(
    tokenize_with_eos,
    batched=True,
    num_proc=max_cpu_count,
    remove_columns=ds["train"].column_names,
)


def pack_dataset(dataset, context_length):
    # 모든 'input_ids' 리스트를 하나의 거대한 리스트로 결합
    all_tokens = []
    for example in tqdm(dataset["input_ids"], desc="Flattening input_ids"):
        all_tokens.extend(example)  # 각 문서(EOS 포함)를 순서대로 추가

    print(
        f"데이터셋의 총 토큰 수: {len(all_tokens):,} "
        f"({len(all_tokens)/1_000_000_000:.4f}B, {len(all_tokens)/1_000_000_000_000:.4f}T)"
    )

    packed_examples = {"input_ids": []}
    for i in tqdm(
        range(0, len(all_tokens) // context_length * context_length, context_length),
        desc="Packing dataset",
        unit="chunk",
    ):
        chunk = all_tokens[i : i + context_length]
        packed_examples["input_ids"].append(chunk)

    return Dataset.from_dict(packed_examples)


print("\n데이터셋을 패킹 중...")
train_packed = pack_dataset(tokenized_ds["train"], context_length)
test_packed = pack_dataset(tokenized_ds["test"], context_length)

tokenized_dataset = {"train": train_packed, "test": test_packed}

print("\n패킹 완료된 데이터셋 구조:")
print(tokenized_dataset)
print(f"훈련 샘플 수: {len(tokenized_dataset['train'])}")
print(f"테스트 샘플 수: {len(tokenized_dataset['test'])}")
print(f"샘플 0의 토큰 수: {len(tokenized_dataset['train'][0]['input_ids'])}")
print(f"샘플 0의 마지막 5개 토큰: {tokenized_dataset['train'][0]['input_ids'][-5:]}")

config = LlamaConfig(
    hidden_size=480,
    num_hidden_layers=32,
    intermediate_size=1920,
    tie_word_embeddings=True,
    num_attention_heads=6,
    num_key_value_heads=2,
    vocab_size=len(tokenizer),
    max_position_embeddings=context_length,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    rope_theta=10000.0,
    use_cache=False,
)


config._attn_implementation = "flash_attention_2"
model = LlamaForCausalLM(config)
model = model.to(torch.bfloat16)

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**3:.2f}B parameters")


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

hf_model_id = "minpeter/tiny-ko-124m-base"
local_model_path = "model/tiny-ko-124m-base"

tokenizer.save_pretrained(local_model_path)
tokenizer.push_to_hub(hf_model_id)

args = TrainingArguments(
    output_dir=local_model_path,
    push_to_hub=True,
    hub_model_id=hf_model_id,
    hub_strategy="every_save",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1_000,
    save_steps=1_000,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=36,
    per_device_eval_batch_size=36,
    logging_steps=25,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    learning_rate=2e-3,
    optim="adamw_torch_fused",
    dataloader_pin_memory=True,
    bf16=True,
    torch_compile=True,
    dataloader_num_workers=max_cpu_count,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()
