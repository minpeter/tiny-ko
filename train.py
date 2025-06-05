import os
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

ds = load_dataset('minpeter/pretrain-korean-dedup', split='train')
ds = ds.train_test_split(test_size=0.001, shuffle=True, seed=5768112)
print(ds)

context_length = 2048
max_cpu_count = int(os.cpu_count() / 3) or 1

tokenizer = AutoTokenizer.from_pretrained("kakaocorp/kanana-nano-2.1b-base")

if tokenizer.eos_token is None:
    raise ValueError("토크나이저에 EOS 토큰이 정의되어 있지 않습니다. 모델을 학습하기 전에 EOS 토큰을 설정해야 합니다.")
if tokenizer.pad_token is None:
    print("토크나이저에 PAD 토큰이 정의되어 있지 않아 EOS 토큰을 PAD 토큰으로 사용합니다.")
    tokenizer.pad_token = tokenizer.eos_token

print(f"사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}") # 이제 EOS 토큰과 동일하거나, 원래 PAD 토큰 값

def append_eos_to_text(examples):
    processed_texts = [text + tokenizer.eos_token for text in examples['text']]
    examples['text'] = processed_texts
    return examples

print("\n각 문서에 EOS 토큰 추가 중...")
ds_with_eos = ds.map(
    append_eos_to_text,
    batched=True,
    batch_size=5_000,
    num_proc=max_cpu_count
)
print("EOS 추가 후 데이터셋 샘플 (text 필드만):")
print(ds_with_eos["train"][0]['text'][-100:])

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
    last_tokens = tokenized_dataset["train"][i]['input_ids'][-5:]
    print(f"샘플 {i}의 마지막 5개 토큰 ID: {last_tokens}, EOS ID와 비교: {tokenizer.eos_token_id}")
    if tokenizer.eos_token_id in last_tokens:
        print(f"  샘플 {i}의 마지막에 EOS 토큰 ID({tokenizer.eos_token_id})가 포함되어 있습니다.")


# 🚀 모델 초기화 (vocab_size는 토크나이저 길이에 맞춤)
config = AutoConfig.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    vocab_size=len(tokenizer),    # EOS 또는 다른 토큰 추가로 인해 tokenizer 길이가 변경되었을 수 있음
    max_position_embeddings=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id, # 위에서 설정된 tokenizer.eos_token_id 사용
    pad_token_id=tokenizer.pad_token_id, # 위에서 설정된 tokenizer.pad_token_id 사용 (eos_token_id와 같을 수 있음)
)

model = LlamaForCausalLM(config)
# 모델 config에도 pad_token_id를 명시적으로 다시 한번 설정 (config 객체와 model.config가 동일 참조가 아닐 수 있으므로)
model.config.pad_token_id = tokenizer.pad_token_id

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**3:.1f}B parameters")

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

hf_model_id = "minpeter/pretrained-tiny-ko"
local_model_path = "model/pretrained-tiny-ko"

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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    logging_steps=5,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    bf16=True,
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