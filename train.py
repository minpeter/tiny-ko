# RUN COMMAND: time uv run accelerate launch train.py

import os
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    LlamaConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets

ds_kr = load_dataset("minpeter/tiny-ko-corpus", split="train")

# >>> en dataset >>>
cosmopedia = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    data_files=[f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(21)],
    split="train",
)
fineweb = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    data_files=[f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet" for i in range(21)],
    split="train",
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

context_length = 8192
max_cpu_count = int(os.cpu_count() / 3) or 1

tokenizer = AutoTokenizer.from_pretrained("./tknz/tiny-ko-tokenizer")

try:
    print(f"사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
except AttributeError as e:
    print(e)


def append_eos_to_text(examples):
    processed_texts = [text + tokenizer.eos_token for text in examples["text"]]
    examples["text"] = processed_texts
    return examples


print("\n각 문서에 EOS 토큰 추가 중...")
ds_with_eos = ds.map(
    append_eos_to_text, batched=True, batch_size=5_000, num_proc=max_cpu_count
)
print("EOS 추가 후 데이터셋 샘플 (text 필드만):")
print(ds_with_eos["train"][0]["text"][-100:])


def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    return outputs


print("\n토큰화 진행 중...")
tokenized_dataset = ds_with_eos.map(
    tokenize,
    remove_columns=ds_with_eos["train"].column_names,
    batched=True,
    batch_size=5_000,
    num_proc=max_cpu_count,
)
print("토큰화된 데이터셋 구조:")
print(tokenized_dataset)
print("토큰화된 데이터셋 샘플 (input_ids):")
for i in range(min(5, len(tokenized_dataset["train"]))):
    last_tokens = tokenized_dataset["train"][i]["input_ids"][-5:]
    print(
        f"샘플 {i}의 마지막 5개 토큰 ID: {last_tokens}, EOS ID와 비교: {tokenizer.eos_token_id}"
    )
    if tokenizer.eos_token_id in last_tokens:
        print(
            f"  샘플 {i}의 마지막에 EOS 토큰 ID({tokenizer.eos_token_id})가 포함되어 있습니다."
        )


# 🚀 모델 초기화 (vocab_size는 토크나이저 길이에 맞춤)
# config = AutoConfig.from_pretrained(
#     "HuggingFaceTB/SmolLM2-135M",
#     vocab_size=len(tokenizer),    # EOS 또는 다른 토큰 추가로 인해 tokenizer 길이가 변경되었을 수 있음
#     max_position_embeddings=context_length,
#     # bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id, # 위에서 설정된 tokenizer.eos_token_id 사용
#     pad_token_id=tokenizer.pad_token_id, # 위에서 설정된 tokenizer.pad_token_id 사용 (eos_token_id와 같을 수 있음)
# )
config = LlamaConfig(
    hidden_size=256,
    num_hidden_layers=12,
    intermediate_size=1024,
    tie_word_embeddings=True,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=len(tokenizer),
    max_position_embeddings=context_length,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


config._attn_implementation = "flash_attention_2"
model = LlamaForCausalLM(config)
model = model.to(torch.bfloat16)

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**3:.2f}B parameters")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, axis=-1)


metric = evaluate.load("accuracy")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

hf_model_id = "minpeter/tiny-ko-20m-base-en"
local_model_path = "model/tiny-ko-20m-base-en"

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
    gradient_accumulation_steps=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=5,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    bf16=True,
    torch_compile=True,
    dataloader_num_workers=max_cpu_count,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
